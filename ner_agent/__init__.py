import pathlib
import types
from enum import StrEnum

import agents
import openai
import pydantic
from openai.types import ChatModel
from str_or_none import str_or_none

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()

DEFAULT_MODEL = "gpt-4.1-nano"


class EntityType(StrEnum):
    """
    A simplified set of 12 entity types, consolidated from spaCy's standard 18.
    """

    PERSON = "PERSON"
    NORP = "NORP"
    FAC = "FAC"
    ORG = "ORG"
    LOCATION = "LOCATION"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    DATETIME = "DATETIME"
    NUMERIC = "NUMERIC"


entity_descriptions = types.MappingProxyType(
    {
        EntityType.PERSON: "People, including fictional characters. Ex: 'Elon Musk', 'Zhuge Liang'",  # noqa: E501
        EntityType.NORP: "Nationalities or religious or political groups. Ex: 'Taiwanese', 'Buddhist', 'Republican'",  # noqa: E501
        EntityType.FAC: "Buildings, airports, highways, bridges, etc. Ex: 'Taipei 101', 'JFK Airport'",  # noqa: E501
        EntityType.ORG: "Companies, agencies, institutions, etc. Ex: 'Google', 'TSMC', 'United Nations'",  # noqa: E501
        EntityType.LOCATION: "Geopolitical entities and other locations like mountains or water bodies. Ex: 'Taiwan', 'New York City', 'Mount Everest'",  # noqa: E501
        EntityType.PRODUCT: "Objects, vehicles, foods, etc. (not services). Ex: 'iPhone 15', 'Tesla Model S'",  # noqa: E501
        EntityType.EVENT: "Named hurricanes, battles, wars, sports events. Ex: 'Hurricane Katrina', 'Olympics'",  # noqa: E501
        EntityType.WORK_OF_ART: "Titles of books, songs, works of art, etc. Ex: 'Mona Lisa', 'The Lord of the Rings'",  # noqa: E501
        EntityType.LAW: "Named documents made into laws. Ex: 'The Constitution', 'Title IX'",  # noqa: E501
        EntityType.LANGUAGE: "Any named language. Ex: 'Chinese', 'English', 'Python' (the language)",  # noqa: E501
        EntityType.DATETIME: "Absolute or relative dates, times, or periods. Ex: 'July 22, 2025', 'yesterday', '9:30 AM'",  # noqa: E501
        EntityType.NUMERIC: "All numerical types including money, quantity, percentages, ordinals, and cardinals. Ex: '20%', '$100', '10 kg', 'first', 'one thousand'",  # noqa: E501
    }
)


class NerAgent:
    async def run(
        self,
        text: str,
        *,
        model: (
            agents.OpenAIChatCompletionsModel
            | agents.OpenAIResponsesModel
            | ChatModel
            | str
            | None
        ) = None,
        tracing_disabled: bool = True,
        **kwargs,
    ) -> "NerResult":
        if str_or_none(text) is None:
            raise ValueError("text is required")

        chat_model = self._to_chat_model(model)

        agent = agents.Agent(name="ner-agent", model=chat_model)
        result = await agents.Runner.run(
            agent, text, run_config=agents.RunConfig(tracing_disabled=tracing_disabled)
        )
        return NerResult()

    def _to_chat_model(
        self,
        model: (
            agents.OpenAIChatCompletionsModel
            | agents.OpenAIResponsesModel
            | ChatModel
            | str
            | None
        ) = None,
    ) -> agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel:
        model = DEFAULT_MODEL if model is None else model

        if isinstance(model, str):
            openai_client = openai.AsyncOpenAI()
            return agents.OpenAIResponsesModel(
                model=model,
                openai_client=openai_client,
            )

        else:
            return model


class Entity(pydantic.BaseModel):
    name: str
    value: str
    start: int = 0
    end: int = 0


class NerResult(pydantic.BaseModel):
    text: str
    entities: list[Entity] = pydantic.Field(default_factory=list)
