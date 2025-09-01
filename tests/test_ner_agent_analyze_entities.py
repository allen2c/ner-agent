# tests/test_ner_agent_simple_entities.py
import agents
import pytest

from ner_agent import NerAgent

TEST_CASES: list[tuple[str, list[str]]] = [
    ("Apple published their first iPhone in 2007.", ["Apple", "iPhone", "2007"]),
    (
        "Microsoft CEO said Windows 10 is the best operating system.",
        ["Microsoft", "Windows 10"],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("text,entities", TEST_CASES)
async def test_ner_agent_analyze_entities(
    text: str,
    entities: list[str],
    chat_model: agents.OpenAIChatCompletionsModel,
):
    agent = NerAgent()
    result = await agent.analyze_entities(text, model=chat_model, verbose=True)
    assert set([e.value for e in result.entities]) >= set(entities)
