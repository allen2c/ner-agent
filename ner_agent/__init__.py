import pathlib
import re
import textwrap
import types
from enum import StrEnum

import agents
import jinja2
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
        EntityType.DATETIME: "Absolute or relative dates, times, or periods. Ex: 'July 22, 2025', 'yesterday', '9:30 AM'",  # noqa: E501
        EntityType.NUMERIC: "All numerical types including money, quantity, percentages, ordinals, and cardinals. Ex: '20%', '$100', '10 kg', 'first', 'one thousand'",  # noqa: E501
    }
)


class NerAgent:
    instructions: str = textwrap.dedent(
        """
        Your task is to do named entity recognition (NER) on the given text.
        The entity format is in [ENTITY_TEXT](#ENTITY_TYPE) and seperated by pipe "|", e.g. [Apple](#ORG)|[Cupertino](#GPE)|[Tim Cook](#PERSON)

        # Entity Definitions
        {% for entity_type, entity_description in entity_descriptions.items() %}
        - {{ entity_type }}: {{ entity_description }}
        {% endfor %}

        # Examples

        text: "Elon Musk visited Tesla's Gigafactory in Austin on March 15, 2024, and announced a 20% increase in production capacity."
        entities: [Elon Musk](#PERSON)|[Tesla](#ORG)|[Gigafactory](#FAC)|[Austin](#LOCATION)|[March 15, 2024](#DATETIME)|[20%](#NUMERIC)

        text: "La presidenta mexicana visitó la sede de las Naciones Unidas en Nueva York el martes pasado para discutir los derechos humanos."
        entities: [mexicana](#NORP)|[Naciones Unidas](#ORG)|[Nueva York](#LOCATION)|[martes pasado](#DATETIME)|[derechos humanos](#LAW)

        text: "蘋果公司在台北101發表了iPhone 15，預計售價為新台幣35,000元"
        entities: [蘋果公司](#ORG)|[台北101](#FAC)|[iPhone 15](#PRODUCT)|[新台幣35,000元](#NUMERIC)

        text: "東京オリンピックで日本人選手が金メダルを獲得し、君が代が演奏された。"
        entities: [東京オリンピック](#EVENT)|[日本人](#NORP)|[金メダル](#PRODUCT)|[君が代](#WORK_OF_ART)

        text: "삼성전자는 서울 강남구에서 오전 9시에 갤럭시 S24를 공개했고, 한국어 AI 기능을 강조했다."
        entities: [삼성전자](#ORG)|[서울](#LOCATION)|[강남구](#LOCATION)|[오전 9시](#DATETIME)|[갤럭시 S24](#PRODUCT)|[한국어](#LANGUAGE)

        text: "The Buddhist monks from Mount Fuji will perform at Carnegie Hall next Friday, celebrating the first anniversary of their Peace Treaty."
        entities: [Buddhist](#NORP)|[Mount Fuji](#LOCATION)|[Carnegie Hall](#FAC)|[next Friday](#DATETIME)|[first](#NUMERIC)|[Peace Treaty](#LAW)

        # Input

        text: {{ text }}
        entities:
        """  # noqa: E501
    )

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

        agent = agents.Agent(
            name="ner-agent",
            model=chat_model,
            instructions=jinja2.Template(self.instructions).render(
                text=text,
                entity_descriptions=entity_descriptions,
            ),
        )
        result = await agents.Runner.run(
            agent, text, run_config=agents.RunConfig(tracing_disabled=tracing_disabled)
        )
        return NerResult(
            text=text,
            entities=self._parse_entities(str(result.final_output)),
        )

    def _parse_entities(
        self, entity_string: str, original_text: str = ""
    ) -> list["Entity"]:
        """
        Parse entities from the format: [ENTITY_TEXT](#ENTITY_TYPE)|[ENTITY_TEXT](#ENTITY_TYPE)

        Args:
            entity_string: The entity string to parse (output from the LLM)
            original_text: The original input text to find entity positions in
        """  # noqa: E501
        entities = []

        # Pattern to match [text](#type) format
        pattern = r"\[([^\]]+)\]\(#([^)]+)\)"

        # Split by pipe and process each entity
        entity_parts = entity_string.strip().split("|")

        for part in entity_parts:
            part = part.strip()
            if not part:
                continue

            match = re.match(pattern, part)
            if match:
                entity_text = match.group(1)
                entity_type = match.group(2)

                # Find the position of this entity in the original text
                start_pos = original_text.find(entity_text) if original_text else 0
                end_pos = start_pos + len(entity_text) if start_pos != -1 else 0

                entities.append(
                    Entity(
                        name=entity_type,
                        value=entity_text,
                        start=start_pos,
                        end=end_pos,
                    )
                )

        return entities

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
