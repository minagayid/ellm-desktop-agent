from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training" / "data"
CANDIDATE_ROOT = ROOT / "models" / "candidates"
BASE_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
BASE_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
ADAPTER_ID = "minagayid/ELLM"
ADAPTER_REVISION = "3317030574fe5564454573efd8150e4e7178c033"
SEED = 20260930
MAX_LENGTH = 1024
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 5e-5


class EncodedConversations(Dataset):
    def __init__(self, path: Path, tokenizer):
        self.items: list[tuple[list[int], list[int]]] = []
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            messages = row["messages"]
            prompt = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            full = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            if full[: len(prompt)] != prompt:
                raise ValueError(f"chat template does not preserve the prompt prefix for {row['id']}")
            if len(full) > MAX_LENGTH:
                raise ValueError(f"example {row['id']} exceeds {MAX_LENGTH} tokens")
            labels = [-100] * len(prompt) + full[len(prompt) :]
            if not any(token != -100 for token in labels):
                raise ValueError(f"no assistant target tokens for {row['id']}")
            self.items.append((full, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.items[index]


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(input_ids) for input_ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for index, (tokens, targets) in enumerate(batch):
        input_ids[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        attention[index, : len(tokens)] = 1
        labels[index, : len(targets)] = torch.tensor(targets, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def mean_loss(model, data: Dataset, device: torch.device, dtype: torch.dtype) -> float:
    pad_id = model.config.pad_token_id or model.config.eos_token_id
    loader = DataLoader(data, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate(batch, pad_id))
    model.eval()
    total_loss = 0.0
    examples = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=dtype):
                loss = model(**batch).loss
            total_loss += float(loss) * len(batch["input_ids"])
            examples += len(batch["input_ids"])
    model.train()
    return total_loss / max(examples, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue ELLM's LoRA adapter on a small bilingual task curriculum.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=CANDIDATE_ROOT / "latest")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This bounded training run requires a local CUDA GPU. No remote/paid fallback is used.")
    if not 1 <= args.epochs <= 3:
        raise SystemExit("Choose 1–3 epochs; use the held-out evaluation before selecting a candidate.")
    candidate = args.output_dir.resolve()
    if candidate.parent != CANDIDATE_ROOT.resolve():
        raise SystemExit(f"Candidate must be a direct child of {CANDIDATE_ROOT}")
    if candidate.exists():
        raise SystemExit(f"Refusing to overwrite candidate: {candidate}")
    for split in ("train", "dev"):
        if not (args.data_dir / f"{split}.jsonl").is_file():
            raise SystemExit(f"Missing {split}.jsonl. Build the dataset first.")
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Missing dataset manifest. Build the dataset first.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    print(f"Loading {ADAPTER_ID}@{ADAPTER_REVISION} locally; no data will be uploaded.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_ID, revision=BASE_REVISION, trust_remote_code=False, use_fast=True
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("base tokenizer has no EOS token")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        revision=BASE_REVISION,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(
        base, ADAPTER_ID, revision=ADAPTER_REVISION, is_trainable=True
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.eos_token_id
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if not trainable_count:
        raise RuntimeError("ELLM adapter loaded without trainable parameters")

    train_data = EncodedConversations(args.data_dir / "train.jsonl", tokenizer)
    dev_data = EncodedConversations(args.data_dir / "dev.jsonl", tokenizer)
    if not len(train_data) or not len(dev_data):
        raise ValueError("train and dev splits must both be nonempty")
    train_loader = DataLoader(
        train_data,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda batch: collate(batch, tokenizer.eos_token_id),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    started = time.perf_counter()
    train_losses: list[float] = []
    dev_losses: list[float] = []
    best_dev_loss = float("inf")
    selected_epoch = 0
    best_trainable_state: dict[str, torch.Tensor] = {}
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=dtype):
                loss = model(**batch).loss
            running_loss += float(loss.detach())
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_loss = running_loss / max(len(train_loader), 1)
        dev_loss = mean_loss(model, dev_data, device, dtype)
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            selected_epoch = epoch + 1
            best_trainable_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        print(json.dumps({"epoch": epoch + 1, "train_loss": train_loss, "dev_loss": dev_loss}), flush=True)

    if not best_trainable_state:
        raise RuntimeError("training completed without a selectable development checkpoint")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_trainable_state[name].to(device=parameter.device, dtype=parameter.dtype))

    elapsed = time.perf_counter() - started
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": BASE_ID,
        "base_model_revision": BASE_REVISION,
        "starting_adapter_id": ADAPTER_ID,
        "starting_adapter_revision": ADAPTER_REVISION,
        "training_data_manifest_sha256": sha256(manifest_path),
        "training_rows": len(train_data),
        "dev_rows": len(dev_data),
        "seed": args.seed,
        "epochs": args.epochs,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "trainable_parameters": trainable_count,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "train_loss_by_epoch": train_losses,
        "dev_loss_by_epoch": dev_losses,
        "selected_epoch": selected_epoch,
        "selected_dev_loss": best_dev_loss,
        "elapsed_seconds": elapsed,
        "dataset_splits": data_manifest.get("splits", {}),
        "upload_performed": False,
        "status": "candidate_not_promoted",
    }
    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".candidate-", dir=CANDIDATE_ROOT) as staging_name:
        staging = Path(staging_name)
        model.save_pretrained(staging, safe_serialization=True)
        (staging / "training-run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staging.rename(candidate)
    print(json.dumps({"candidate": str(candidate), "selected_epoch": selected_epoch, "selected_dev_loss": best_dev_loss, "train_losses": train_losses, "dev_losses": dev_losses, "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
