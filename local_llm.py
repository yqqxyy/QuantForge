from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = os.environ.get("QUANTFORGE_LLM_MODEL", "mlx-community/Qwen3.5-4B-OptiQ-4bit")
DEFAULT_HF_HOME = Path(os.environ.get("QUANTFORGE_HF_HOME", Path.home() / ".cache" / "huggingface"))


@dataclass(frozen=True)
class LocalLLMConfig:
    model: str = DEFAULT_MODEL
    hf_home: Path = DEFAULT_HF_HOME
    max_tokens: int = 900


class LocalQwenMLX:
    def __init__(self, config: LocalLLMConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        os.environ.setdefault("HF_HOME", str(self.config.hf_home))
        from mlx_lm import load

        self._model, self._tokenizer = load(self.config.model)

    def generate(self, prompt: str) -> str:
        self.load()
        from mlx_lm import generate

        return generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
        )


def build_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
