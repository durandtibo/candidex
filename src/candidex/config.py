r"""Contain configuration classes."""

from __future__ import annotations

__all__ = ["MAX_RETRIES_DEFAULT", "ChatModelConfig"]

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

MAX_RETRIES_DEFAULT = 9999
"""Default maximum number of retries on failed LLM calls.

Set deliberately high so that transient network or rate-limit errors are
retried until they resolve, rather than aborting a long-running batch
job. Override with a lower value in latency-sensitive or cost-sensitive
contexts.
"""

@dataclass(frozen=True)
class ChatModelConfig:
    r"""A generic LLM configuration.

    Attributes:
        model:         The model identifier string passed to ``init_chat_model``
                       (e.g. ``"openai:gpt-4o"`` or ``"ollama:gemma3:1b"``).
        system_prompt: The system prompt that instructs the LLM on its
                       role and task.
        batch_size:    Number of examples to process concurrently per
                       inference batch. Defaults to ``1``.
        max_retries:   Maximum number of retries on failed LLM calls.
                       Defaults to ``MAX_RETRIES_DEFAULT`` (9999). Set
                       deliberately high to survive transient failures in
                       long-running batch jobs.
        temperature:   Sampling temperature passed to the LLM. Set to
                       ``0.0`` for deterministic outputs. Defaults to ``0.0``.
        init_kwargs:   Optional extra keyword arguments forwarded to
                       ``init_chat_model``. Defaults to ``None``.
    """

    model: str
    system_prompt: str
    batch_size: int = 1
    max_retries: int = field(default=MAX_RETRIES_DEFAULT)
    temperature: float = 0.0
    init_kwargs: dict[str, Any] | None = field(default=None)

    def hash(self) -> str:
        """Return a stable SHA-256 hex digest of the current
        configuration.

        Serialises all attributes to a canonical JSON string with sorted keys
        before hashing, ensuring that two configs with identical values always
        produce the same hash regardless of dict key insertion order in
        ``init_kwargs``.

        Useful for cache keys, output filenames, or detecting configuration
        changes between runs without comparing each field manually.

        Returns:
            A 64-character lowercase hexadecimal SHA-256 digest string.

        Example:
            >>> config = ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful.")
            >>> config.hash()
        """
        canonical = json.dumps(
            {
                "model": self.model,
                "system_prompt": self.system_prompt,
                "batch_size": self.batch_size,
                "max_retries": self.max_retries,
                "temperature": self.temperature,
                "init_kwargs": self.init_kwargs,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()
