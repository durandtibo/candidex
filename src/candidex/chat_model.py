r"""Contain utility functions for chat models."""

from __future__ import annotations

__all__ = ["create_chat_model"]

import logging
from typing import TYPE_CHECKING

from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from candidex.config import ChatModelConfig

logger: logging.Logger = logging.getLogger(__name__)


def create_chat_model(config: ChatModelConfig) -> BaseChatModel:
    r"""Instantiate a LangChain chat model from a
    :class:`~candidex.config.ChatModelConfig`.

    Uses :func:`langchain.chat_models.init_chat_model` under the hood,
    forwarding the ``model``, ``temperature``, ``max_retries``, and any
    extra ``init_kwargs`` from the config. Logs the class name, resolved
    model version, and temperature at ``INFO`` level.

    Args:
        config: The chat model configuration specifying the model
            identifier, sampling temperature, retry settings, and any
            optional extra keyword arguments.

    Returns:
        The instantiated :class:`~langchain_core.language_models.BaseChatModel`.
    """
    init_kwargs = config.init_kwargs or {}
    model = init_chat_model(
        model=config.model,
        temperature=config.temperature,
        max_retries=config.max_retries,
        **init_kwargs,
    )
    model_version = getattr(model, "model", getattr(model, "model_name", "Unknown"))
    logger.info(
        f"class: {type(model).__name__} | model: {model_version} | temperature: {model.temperature}"
    )
    return model
