# tests/test_ner_agent_run.py
import typing
from pprint import pformat

import agents
import pytest

from ner_agent import Entities, Entity, EntityType, NerAgent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,entities",
    [
        (
            "My name is John Doe and I live in New York City",
            [
                Entity(name=EntityType.PERSON, value="John Doe", start=11, end=19),
                Entity(
                    name=EntityType.LOCATION, value="New York City", start=34, end=47
                ),
            ],
        ),
    ],
)
async def test_ner_agent_run(
    text: str,
    entities: typing.List[Entity],
    chat_model: agents.OpenAIChatCompletionsModel,
):
    agent = NerAgent()
    result = await agent.run(text, model=chat_model, verbose=True)
    assert pformat(Entities.dump_python(result.entities)) == pformat(
        Entities.dump_python(entities)
    )
