# tests/test_ner_agent_simple_entities.py
import agents
import pytest

from ner_agent import NerAgent

TEST_CASES: list[tuple[list[str], bool, str | None]] = [
    (["Apple", "Microsoft", "Nvidia"], False, None),
    (["黃仁勳", "Jensen Huang", "jensenhuang"], True, "Jensen Huang"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("texts, is_synonymous, canonical_name", TEST_CASES)
async def test_ner_agent_analyze_synonyms_and_canonical_name(
    texts: list[str],
    is_synonymous: bool,
    canonical_name: str | None,
    chat_model: agents.OpenAIChatCompletionsModel,
):
    agent = NerAgent()
    result = await agent.analyze_synonyms_and_canonical_name(
        texts, model=chat_model, verbose=True
    )
    print(result)
    assert result.is_synonymous == is_synonymous
    assert result.canonical_name == canonical_name
