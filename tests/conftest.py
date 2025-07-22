import agents
import openai
import pytest


@pytest.fixture(scope="module")
def chat_model_str():
    return "gemma3n:e4b"


@pytest.fixture(scope="module")
def chat_model(chat_model_str: str):
    client = openai.AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    return agents.OpenAIChatCompletionsModel(model=chat_model_str, openai_client=client)
