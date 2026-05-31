import pytest

from candidex.config import MAX_RETRIES_DEFAULT, ChatModelConfig

# --- Fixtures ---


@pytest.fixture
def base_config() -> ChatModelConfig:
    return ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful.")


#####################################
#     Tests for ChatModelConfig     #
#####################################

# --- Defaults ---


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        pytest.param("batch_size", 1, id="batch_size"),
        pytest.param("max_retries", MAX_RETRIES_DEFAULT, id="max_retries"),
        pytest.param("temperature", 0.0, id="temperature"),
        pytest.param("init_kwargs", None, id="init_kwargs"),
    ],
)
def test_chat_model_config_default_values(
    base_config: ChatModelConfig, field: str, expected: object
) -> None:
    assert getattr(base_config, field) == expected


# --- Hash format ---


def test_chat_model_config_hash_is_64_char_lowercase_hex(base_config: ChatModelConfig) -> None:
    digest = base_config.hash()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# --- Hash stability ---


def test_chat_model_config_hash_same_config_same_hash() -> None:
    a = ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful.")
    b = ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful.")
    assert a.hash() == b.hash()


def test_chat_model_config_hash_init_kwargs_key_order_does_not_affect_hash() -> None:
    a = ChatModelConfig(
        model="openai:gpt-4o",
        system_prompt="You are helpful.",
        init_kwargs={"timeout": 30, "api_key": "test"},
    )
    b = ChatModelConfig(
        model="openai:gpt-4o",
        system_prompt="You are helpful.",
        init_kwargs={"api_key": "test", "timeout": 30},
    )
    assert a.hash() == b.hash()


# --- Hash sensitivity ---


@pytest.mark.parametrize(
    ("config_a", "config_b"),
    [
        pytest.param(
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful."),
            ChatModelConfig(
                model="anthropic:claude-3-5-sonnet-20241022", system_prompt="You are helpful."
            ),
            id="model",
        ),
        pytest.param(
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful."),
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are strict."),
            id="system_prompt",
        ),
        pytest.param(
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful."),
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful. "),
            id="system_prompt_trailing_whitespace",
        ),
        pytest.param(
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful."),
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful.", batch_size=4),
            id="batch_size",
        ),
        pytest.param(
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful."),
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful.", max_retries=3),
            id="max_retries",
        ),
        pytest.param(
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful."),
            ChatModelConfig(
                model="openai:gpt-4o", system_prompt="You are helpful.", temperature=0.5
            ),
            id="temperature",
        ),
        pytest.param(
            ChatModelConfig(model="openai:gpt-4o", system_prompt="You are helpful."),
            ChatModelConfig(
                model="openai:gpt-4o", system_prompt="You are helpful.", init_kwargs={"timeout": 30}
            ),
            id="init_kwargs_none_vs_value",
        ),
        pytest.param(
            ChatModelConfig(
                model="openai:gpt-4o", system_prompt="You are helpful.", init_kwargs={}
            ),
            ChatModelConfig(
                model="openai:gpt-4o", system_prompt="You are helpful.", init_kwargs=None
            ),
            id="init_kwargs_empty_vs_none",
        ),
    ],
)
def test_chat_model_config_hash_different_configs_have_different_hashes(
    config_a: ChatModelConfig, config_b: ChatModelConfig
) -> None:
    assert config_a.hash() != config_b.hash()
