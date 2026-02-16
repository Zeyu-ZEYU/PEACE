"""
Prompt construction utilities.

Azure trace does not include raw prompt text for privacy reasons; it provides
only token counts. To run *real* inference experiments, we must synthesize a
prompt string with a target token length under the model's tokenizer.

This module generates prompts by repeating a single token ID N times and
decoding it back to text, which yields an exact token count under the same
tokenizer (excluding any server-side special token additions).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence


class TokenizerNotAvailable(RuntimeError):
    """Raised when transformers/tokenizer cannot be imported."""


def _load_tokenizer(tokenizer_name_or_path: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise TokenizerNotAvailable(
            "transformers is required to build token-accurate prompts. "
            "Install it with: pip install transformers"
        ) from e

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    return tok


def _find_single_token_id(tokenizer) -> int:
    """
    Find a token ID that encodes from a short string into exactly 1 token.

    This is used to build token-accurate long prompts efficiently.
    """
    candidates = [
        " the",
        " a",
        " hello",
        " world",
        ".",
        ",",
        "I",
        " you",
        "0",
        "1",
        "A",
        "B",
    ]
    for s in candidates:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in getattr(tokenizer, "all_special_ids", []):
            return int(ids[0])

    # Fallback: scan token IDs until we find a non-empty, non-special decode.
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for tid in range(vocab_size):
        if tid in special:
            continue
        text = tokenizer.decode([tid], skip_special_tokens=True)
        if text and text.strip():
            return int(tid)

    raise RuntimeError("Unable to find a usable single token ID for this tokenizer.")


@dataclass
class PromptBuilder:
    """
    Build synthetic prompts with a target token length under a tokenizer.
    """
    tokenizer_name_or_path: str
    prefix: str = ""
    suffix: str = ""

    def __post_init__(self) -> None:
        self.tokenizer = _load_tokenizer(self.tokenizer_name_or_path)
        self._fill_token_id = _find_single_token_id(self.tokenizer)

        self._prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False) if self.prefix else []
        self._suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False) if self.suffix else []

    @lru_cache(maxsize=4096)
    def build(self, target_tokens: int) -> str:
        """
        Build a prompt string with an *exact* token length equal to target_tokens
        under this tokenizer (assuming add_special_tokens=False).
        """
        if target_tokens <= 0:
            return ""

        reserved = len(self._prefix_ids) + len(self._suffix_ids)
        if reserved > target_tokens:
            raise ValueError(
                f"Prefix+suffix consume {reserved} tokens, exceeds target_tokens={target_tokens}."
            )

        fill_len = target_tokens - reserved
        token_ids = list(self._prefix_ids) + [self._fill_token_id] * fill_len + list(self._suffix_ids)
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text
