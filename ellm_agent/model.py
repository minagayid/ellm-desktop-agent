from __future__ import annotations

from pathlib import Path
from typing import Any

from .limits import bounded_setting


class LocalELLM:
    def __init__(self, config: dict[str, Any], adapter_override: str | None = None):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Model packages are missing. Run ./install.ps1 first.") from exc

        self.torch = torch
        self.adapter_id = adapter_override or config["adapter_id"]
        self.adapter_revision = None if adapter_override else config.get("adapter_revision")
        base_id = config["base_model_id"]
        base_revision = config.get("base_model_revision")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        print(f"Loading local ELLM from {self.adapter_id} (first run downloads model files)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_id,
            revision=base_revision,
            trust_remote_code=False,
            use_fast=True,
        )
        load_args: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": False,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
        base = AutoModelForCausalLM.from_pretrained(base_id, revision=base_revision, **load_args)
        adapter_args: dict[str, Any] = {"is_trainable": False}
        if self.adapter_revision:
            adapter_args["revision"] = self.adapter_revision
        self.model = PeftModel.from_pretrained(base, self.adapter_id, **adapter_args)
        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True
        self.max_new_tokens = bounded_setting(config, "max_new_tokens", 256, 512)
        self.max_input_tokens = bounded_setting(config, "max_input_tokens", 4_096, 6_144)

    def generate(self, messages: list[dict[str, str]]) -> str:
        torch = self.torch
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        ).to(self.device)
        if encoded.shape[1] > self.max_input_tokens:
            raise ValueError(
                f"conversation is {encoded.shape[1]} tokens; this run is limited to {self.max_input_tokens}. "
                "Shorten the request or reduce the number of files in the task."
            )
        with torch.inference_mode():
            generated = self.model.generate(
                encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        answer = generated[0, encoded.shape[1] :]
        return self.tokenizer.decode(answer, skip_special_tokens=True).strip()


def load_model_config(project_root: Path) -> dict[str, Any]:
    import json

    local_path = project_root / "config" / "model.local.json"
    example_path = project_root / "config" / "model.example.json"
    source = local_path if local_path.is_file() else example_path
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model config must be a JSON object")
    required = {"adapter_id", "base_model_id", "base_model_revision"}
    if not required.issubset(value):
        raise ValueError(f"model config is missing {sorted(required - set(value))}")
    if not all(isinstance(value[name], str) and value[name].strip() for name in required):
        raise ValueError("model IDs and revisions must be non-empty strings")
    if "adapter_revision" in value and not isinstance(value["adapter_revision"], str):
        raise ValueError("adapter_revision must be a string")
    return value
