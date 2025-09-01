# tests/test_ner_agent_simple_entities.py
import agents
import pytest

from ner_agent import NerAgent

TEST_CASES: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Apple published their first iPhone in 2007.",
        [("Apple", "related_to", "iPhone")],
    ),
    (
        "Microsoft CEO said Windows 10 is the best operating system.",
        [
            ("Windows 10", "is_a", "operating system"),
        ],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("text,triplets", TEST_CASES)
async def test_ner_agent_extract_relations(
    text: str,
    triplets: list[tuple[str, str, str]],
    chat_model: agents.OpenAIChatCompletionsModel,
):
    agent = NerAgent()
    result = await agent.extract_relations(text, model=chat_model, verbose=True)
    assert set([(e.subject, e.relation, e.object) for e in result.triplets]) >= set(
        triplets
    )
