from __future__ import annotations

import argparse
import hashlib
import json
import os
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


class LossOnly(torch.nn.Module):
    """Return only the scalar loss so multi-GPU training does not gather full logits."""

    def __init__(self, model: PeftModel, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=self.dtype):
            mean_token_loss = self.model(**batch).loss.float()
        valid_tokens = batch["labels"][:, 1:].ne(-100).sum().to(dtype=torch.float32)
        return torch.stack((mean_token_loss * valid_tokens, valid_tokens))


def mean_loss(model: PeftModel, loss_model: torch.nn.Module, data: Dataset, device: torch.device) -> float:
    pad_id = model.config.pad_token_id or model.config.eos_token_id
    loader = DataLoader(data, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate(batch, pad_id))
    model.eval()
    total_loss = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss_stats = loss_model(**batch)
            total_loss += float(loss_stats[..., 0].sum())
            token_count += int(loss_stats[..., 1].sum())
    model.train()
    return total_loss / max(token_count, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def widen_lora_adapter(model: PeftModel, rank: int, alpha: int) -> dict[str, Any]:
    """Widen a standard LoRA adapter while preserving its initial function."""
    active_adapters = model.active_adapters
    if len(active_adapters) != 1:
        raise ValueError("a single active adapter is required for rank expansion")
    adapter_name = active_adapters[0]
    config = model.peft_config[adapter_name]
    old_rank = int(config.r)
    old_alpha = int(config.lora_alpha)
    if rank < old_rank or rank < 1 or alpha < 1:
        raise ValueError("the requested LoRA rank must be positive and at least the existing rank")
    if getattr(config, "use_dora", False) or getattr(config, "use_rslora", False):
        raise ValueError("rank expansion currently supports standard LoRA only")
    if old_alpha / old_rank != alpha / rank:
        raise ValueError("rank changes must preserve the adapter's original lora_alpha/r scaling")

    updated_modules = 0
    for module in model.modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None or adapter_name not in lora_a or adapter_name not in lora_b:
            continue
        old_a = lora_a[adapter_name]
        old_b = lora_b[adapter_name]
        if not isinstance(old_a, torch.nn.Linear) or not isinstance(old_b, torch.nn.Linear):
            continue
        if old_a.out_features != old_rank or old_b.in_features != old_rank:
            raise ValueError("LoRA module rank does not match the adapter configuration")
        if rank > old_rank:
            new_a = torch.nn.Linear(
                old_a.in_features, rank, bias=old_a.bias is not None,
                device=old_a.weight.device, dtype=old_a.weight.dtype,
            )
            new_b = torch.nn.Linear(
                rank, old_b.out_features, bias=old_b.bias is not None,
                device=old_b.weight.device, dtype=old_b.weight.dtype,
            )
            with torch.no_grad():
                new_a.weight[:old_rank].copy_(old_a.weight)
                new_b.weight.zero_()
                new_b.weight[:, :old_rank].copy_(old_b.weight)
                if new_a.bias is not None:
                    new_a.bias[:old_rank].copy_(old_a.bias)
                    new_a.bias[old_rank:].zero_()
                if new_b.bias is not None:
                    new_b.bias.zero_()
            lora_a[adapter_name] = new_a
            lora_b[adapter_name] = new_b
        module.r[adapter_name] = rank
        module.lora_alpha[adapter_name] = alpha
        module.scaling[adapter_name] = alpha / rank
        updated_modules += 1
    if not updated_modules:
        raise RuntimeError("no standard LoRA linear modules were found to update")

    config.r = rank
    config.lora_alpha = alpha
    return {
        "original_rank": old_rank,
        "original_alpha": old_alpha,
        "selected_rank": rank,
        "selected_alpha": alpha,
        "modules_widened": updated_modules if rank > old_rank else 0,
        "modules_updated": updated_modules,
    }


def last_token_logits(model: PeftModel, token_ids: list[int], device: torch.device) -> torch.Tensor:
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        return model(input_ids=ids, attention_mask=mask).logits[:, -1, :].float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue ELLM's LoRA adapter on a small bilingual task curriculum.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "v2")
    parser.add_argument("--output-dir", type=Path, default=CANDIDATE_ROOT / "latest")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This bounded training run requires a CUDA GPU; the script does not provision one.")
    if not 1 <= args.epochs <= 5:
        raise SystemExit("Choose 1–5 epochs; use the development split to select a checkpoint.")
    if args.lora_rank < 8 or args.lora_alpha < 1 or args.learning_rate <= 0:
        raise SystemExit("LoRA rank must be at least 8, alpha and learning rate must be positive.")
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
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "dev"):
        expected_hash = data_manifest.get("splits", {}).get(split, {}).get("sha256")
        actual_hash = sha256(args.data_dir / f"{split}.jsonl")
        if not expected_hash or actual_hash != expected_hash:
            raise SystemExit(f"{split}.jsonl does not match the dataset manifest.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    print(json.dumps({"visible_gpus": gpu_names}), flush=True)
    print(f"Loading pinned ELLM adapter {ADAPTER_ID}@{ADAPTER_REVISION}.", flush=True)
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
    expansion: dict[str, Any] = {"original_rank": None, "selected_rank": None, "modules_widened": 0}
    equivalence_error: float | None = None
    train_data = EncodedConversations(args.data_dir / "train.jsonl", tokenizer)
    dev_data = EncodedConversations(args.data_dir / "dev.jsonl", tokenizer)
    if not len(train_data) or not len(dev_data):
        raise ValueError("train and dev splits must both be nonempty")
    active_adapters = model.active_adapters
    if len(active_adapters) != 1:
        raise SystemExit("ELLM must load with exactly one active adapter.")
    adapter_name = active_adapters[0]
    current_lora = model.peft_config[adapter_name]
    current_rank = int(current_lora.r)
    current_alpha = int(current_lora.lora_alpha)
    if args.lora_rank < current_rank:
        raise SystemExit("LoRA rank may not shrink the existing ELLM adapter.")
    if args.lora_rank != current_rank or args.lora_alpha != current_alpha:
        model.eval()
        original_logits = last_token_logits(model, train_data[0][0], device)
        expansion = widen_lora_adapter(model, args.lora_rank, args.lora_alpha)
        expanded_logits = last_token_logits(model, train_data[0][0], device)
        equivalence_error = float((original_logits - expanded_logits).abs().max())
        if not torch.allclose(original_logits, expanded_logits, atol=5e-4, rtol=1e-4):
            raise RuntimeError(f"expanded adapter changed initial logits (max abs diff={equivalence_error})")
        print(json.dumps({"adapter_expansion": expansion, "initial_logit_max_abs_diff": equivalence_error}), flush=True)
        model.train()
    else:
        expansion = {
            "original_rank": current_rank,
            "original_alpha": current_alpha,
            "selected_rank": current_rank,
            "selected_alpha": current_alpha,
            "modules_widened": 0,
            "modules_updated": 0,
        }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if not trainable_count:
        raise RuntimeError("ELLM adapter loaded without trainable parameters")

    train_loader = DataLoader(
        train_data,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda batch: collate(batch, tokenizer.eos_token_id),
        num_workers=0,
    )
    loss_model: torch.nn.Module = LossOnly(model, dtype)
    if torch.cuda.device_count() > 1:
        loss_model = torch.nn.DataParallel(loss_model, device_ids=list(range(torch.cuda.device_count())))
        print(f"Using data parallel across {torch.cuda.device_count()} GPUs.", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    started = time.perf_counter()
    train_losses: list[float] = []
    dev_losses: list[float] = []
    best_dev_loss = float("inf")
    selected_epoch = 0
    best_trainable_state: dict[str, torch.Tensor] = {}
    model.train()
    for epoch in range(args.epochs):
        running_loss_sum = 0.0
        running_token_count = 0
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss_stats = loss_model(**batch)
            loss_sum = loss_stats[..., 0].sum()
            token_count = loss_stats[..., 1].sum()
            loss = loss_sum / token_count.clamp_min(1)
            running_loss_sum += float(loss_sum.detach())
            running_token_count += int(token_count.detach())
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
        train_loss = running_loss_sum / max(running_token_count, 1)
        dev_loss = mean_loss(model, loss_model, dev_data, device)
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
    run = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": BASE_ID,
        "base_model_revision": BASE_REVISION,
        "starting_adapter_id": ADAPTER_ID,
        "starting_adapter_revision": ADAPTER_REVISION,
        "training_source_commit": os.environ.get("ELLM_TRAINING_COMMIT"),
        "training_data_manifest_sha256": sha256(manifest_path),
        "training_rows": len(train_data),
        "dev_rows": len(dev_data),
        "seed": args.seed,
        "epochs": args.epochs,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "learning_rate": args.learning_rate,
        "lora_expansion": expansion,
        "initial_logit_max_abs_diff_after_expansion": equivalence_error,
        "max_length": MAX_LENGTH,
        "trainable_parameters": trainable_count,
        "gpu": torch.cuda.get_device_name(0),
        "gpus_used": gpu_names,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "train_loss_by_epoch": train_losses,
        "dev_loss_by_epoch": dev_losses,
        "selected_epoch": selected_epoch,
        "selected_dev_loss": best_dev_loss,
        "elapsed_seconds": elapsed,
        "dataset_splits": data_manifest.get("splits", {}),
        "model_upload_performed": False,
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
