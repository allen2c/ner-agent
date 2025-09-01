# ner_agent/__init__.py
import json
import logging
import pathlib
import re
import textwrap
import types
from dataclasses import asdict
from enum import StrEnum

import agents
import jinja2
import openai
import pydantic
from openai.types import ChatModel
from str_or_none import str_or_none

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-nano"


class EntityType(StrEnum):
    PERSON = "PERSON"
    NORP = "NORP"
    LOCATION = "LOCATION"
    DATETIME = "DATETIME"
    NUMERIC = "NUMERIC"
    PROPER_NOUN = "PROPER_NOUN"


entity_descriptions = types.MappingProxyType(
    {
        EntityType.PERSON: "People, including fictional characters. Ex: 'Elon Musk', 'Zhuge Liang'",  # noqa: E501
        EntityType.NORP: "Nationalities, religious groups, political groups, and languages. Ex: 'Taiwanese', 'Buddhist', 'Republican', '中文'",  # noqa: E501
        EntityType.LOCATION: "Geopolitical entities and physical facilities (buildings, airports, bridges, highways, museums, etc.). Ex: 'Taiwan', 'New York City', 'Taipei 101', 'JFK Airport'",  # noqa: E501
        EntityType.DATETIME: "Absolute or relative dates/times/periods/ages. Ex: 'July 22, 2025', 'yesterday', '9:30 AM', 'Q1 FY2024', '5 years old'",  # noqa: E501
        EntityType.NUMERIC: "Numbers of any kind: money, quantities, percentages, ordinals/cardinals. Ex: '20%', '$100', '10 kg', 'first', '2,345'",  # noqa: E501
        EntityType.PROPER_NOUN: (
            "Named events, works of ART/MEDIA, LAWS/TREATIES, BRANDED/NAMED PRODUCTS OR MODELS, "  # noqa: E501
            "AND ORGANIZATIONS/COMPANIES/AGENCIES/INSTITUTIONS. Ex: 'Hurricane Katrina', 'Mona Lisa', "  # noqa: E501
            "'Title IX', 'World War II', 'iPhone 15', 'Tesla', 'United Nations'"  # noqa: E501
        ),
    }
)
legacy_entity_map = types.MappingProxyType(
    {
        "GPE": EntityType.LOCATION,
        "FAC": EntityType.LOCATION,
        "EVENT": EntityType.PROPER_NOUN,
        "WORK_OF_ART": EntityType.PROPER_NOUN,
        "LAW": EntityType.PROPER_NOUN,
        "LANGUAGE": EntityType.NORP,
        "PRODUCT": EntityType.PROPER_NOUN,
        "ORG": EntityType.PROPER_NOUN,
        "ORGANIZATION": EntityType.PROPER_NOUN,
        "PROP_NOWN": EntityType.PROPER_NOUN,
        "NRP": EntityType.NORP,
    }
)


class NerAgent:
    instructions: str = textwrap.dedent(
        """
        Your task is to perform named entity recognition (NER) on the given text.
        Output format: [ENTITY_TEXT](#ENTITY_TYPE) separated by " | " (pipes).
        Example: [Apple](#PROPER_NOUN) | [Taipei 101](#LOCATION) | [Tim Cook](#PERSON)

        # Entity Definitions
        {% for entity_type, entity_description in entity_descriptions.items() -%}
        - {{ entity_type }}: {{ entity_description }}
        {% endfor %}

        # Examples

        text: '''Elon Musk visited Tesla's Gigafactory in Austin on March 15, 2024, and announced a 20% increase.'''
        entities: [Elon Musk](#PERSON) | [Tesla](#PROPER_NOUN) | [Gigafactory](#LOCATION) | [Austin](#LOCATION) | [March 15, 2024](#DATETIME) | [20%](#NUMERIC) | [done](#DONE)

        text: '''La presidenta mexicana visitó la sede de las Naciones Unidas en Nueva York el martes pasado para discutir los derechos humanos.'''
        entities: [mexicana](#NORP) | [Naciones Unidas](#PROPER_NOUN) | [Nueva York](#LOCATION) | [martes pasado](#DATETIME) | [derechos humanos](#PROPER_NOUN) | [done](#DONE)

        text: '''蘋果公司在台北101發表了iPhone 15，預計售價為新台幣35,000元'''
        entities: [蘋果公司](#PROPER_NOUN) | [台北101](#LOCATION) | [iPhone 15](#PROPER_NOUN) | [新台幣35,000元](#NUMERIC) | [done](#DONE)

        text: '''東京オリンピックで日本人選手が金メダルを獲得し、君が代が演奏された。'''
        entities: [東京オリンピック](#PROPER_NOUN) | [日本人](#NORP) | [金メダル](#PROPER_NOUN) | [君が代](#PROPER_NOUN) | [done](#DONE)

        text: '''삼성전자는 서울 강남구에서 오전 9시에 갤럭시 S24를 공개했고, 한국어 AI 기능을 강조했다.'''
        entities: [삼성전자](#PROPER_NOUN) | [서울](#LOCATION) | [강남구](#LOCATION) | [오전 9시](#DATETIME) | [갤럭시 S24](#PROPER_NOUN) | [한국어](#NORP) | [done](#DONE)

        text: '''The Buddhist monks from Mount Fuji will perform at Carnegie Hall next Friday, celebrating the first anniversary of their Peace Treaty.'''
        entities: [Buddhist](#NORP) | [Mount Fuji](#LOCATION) | [Carnegie Hall](#LOCATION) | [next Friday](#DATETIME) | [first](#NUMERIC) | [Peace Treaty](#PROPER_NOUN) | [done](#DONE)

        text: '''L'Hôpital Saint-Louis est un des hôpitaux de Paris.'''
        entities: [L'Hôpital Saint-Louis](#LOCATION) | [hôpitaux](#LOCATION) | [Paris](#LOCATION) | [done](#DONE)

        # Input

        text: '''{{ text }}'''
        entities:
        """  # noqa: E501
    )

    simple_entities_instructions: str = textwrap.dedent(
        """
        ## ROLE: Named Entity Recognition (NER) Specialist

        Your task is to act as a highly accurate NER system. You will be given a single sentence and you must identify and extract all key, concrete entities from it.

        ## INSTRUCTIONS:
        1.  **Identify Entities**: Extract key nouns and noun phrases that represent specific people, organizations, locations, facilities, cuisines, or important objects.
        2.  **Be Concrete**: Focus on concrete entities. Do NOT extract abstract concepts (e.g., "experience", "style", "spirit"), generic nouns (e.g., "way", "details"), adjectives, or verbs.
        3.  **Output Format**: You MUST return a single JSON object with one key: "entities". The value should be a list of strings, where each string is an extracted entity.
        4.  **Empty Result**: If no concrete entities are found, you MUST return an empty list: {% raw %}{"entities": []}{% endraw %}.
        5.  **Do Not Explain**: Do not add any conversational text or explanations. Only output the JSON object.

        ## EXAMPLES:

        ### Example 1
        Input: "Nvidia published their first GPU in 1999."
        Output:
        {% raw %}{"entities": ["Nvidia", "GPU", "1999"]}{% endraw %}

        ### Example 2
        Input: "台北文華東方酒店座落於台北市繁華商業中心與人文薈萃的敦化北路"
        Output:
        {% raw %}{"entities": ["台北文華東方酒店", "台北市繁華商業中心", "敦化北路"]}{% endraw %}

        ## TASK:

        Input: "{{ fact_text }}"
        Output:
        """  # noqa: E501
    )

    synonyms_and_canonical_name_instructions: str = textwrap.dedent(
        """
        ## ROLE: Synonym and Canonical Name Analyst

        You are an expert linguist and data normalizer. Your task is to analyze a list of potential synonyms and make two critical decisions:
        1.  **Validation**: Determine if ALL items in the list refer to the exact same real-world entity.
        2.  **Assignment**: If they are synonyms, select the best canonical (standard) name for the group.

        ## CRITICAL RULES FOR VALIDATION:
        - To be synonymous, all items MUST refer to the IDENTICAL entity.
        - Items that are merely in the same category (e.g., two different hotels, two different times of day, two different Michelin star ratings) are NOT synonymous.
        - Example of TRUE synonyms: ["Hong Kong", "香港"]
        - Example of FALSE synonyms (same category, different entities): ["台北文華東方酒店", "置地文華東方酒店"]

        ## CRITICAL RULES FOR CANONICAL NAME ASSIGNMENT:
        - **Rule 1 (Completeness):** Choose the most complete and unambiguous name. (e.g., "Antonio Guida" is better than "Guida").
        - **Rule 2 (Language Preference):** Prefer the English name if available.
        - **Rule 3 (Fallback):** If no English name is present, use the most complete name from the available options.

        ## OUTPUT FORMAT:
        - You MUST respond with a single JSON object.
        - The JSON object must have two keys:
        1.  `"is_synonymous"`: a boolean (`true` or `false`).
        2.  `"canonical_name"`: a string (if `is_synonymous` is `true`, otherwise `null`).

        ## EXAMPLES:

        ### Example 1
        Input: `["Chef Richard Ekkebus", "Richard Ekkebus"]`
        Output:
        {
        "is_synonymous": true,
        "canonical_name": "Richard Ekkebus"
        }

        ### Example 2
        Input: `["台北文華東方酒店", "置地文華東方酒店"]`
        Output:
        {
        "is_synonymous": false,
        "canonical_name": null
        }

        ### Example 3
        Input: `["Hong Kong", "香港"]`
        Output:
        {
        "is_synonymous": true,
        "canonical_name": "Hong Kong"
        }

        ### Example 4
        Input: `["一星", "三星"]`
        Output:
        {
        "is_synonymous": false,
        "canonical_name": null
        }

        ## TASK:

        Input: `{{ candidate_list }}`
        Output:
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
        verbose: bool = False,
        **kwargs,
    ) -> "NerResult":
        if str_or_none(text) is None:
            raise ValueError("text is required")

        chat_model = self._to_chat_model(model)

        agent_instructions: str = (
            jinja2.Template(self.instructions)
            .render(
                text=text,
                entity_descriptions=entity_descriptions,
            )
            .strip()
        )

        if verbose:
            print("\n\n--- LLM INSTRUCTIONS ---\n")
            print(agent_instructions)

        agent = agents.Agent(
            name="ner-agent",
            model=chat_model,
            model_settings=agents.ModelSettings(temperature=0.0),
            instructions=agent_instructions,
        )
        result = await agents.Runner.run(
            agent, text, run_config=agents.RunConfig(tracing_disabled=tracing_disabled)
        )

        if verbose:
            print("\n\n--- LLM OUTPUT ---\n")
            print(str(result.final_output))
            print("\n--- LLM USAGE ---\n")
            print(
                "Usage:",
                json.dumps(
                    asdict(result.context_wrapper.usage),
                    ensure_ascii=False,
                    default=str,
                ),
            )

        return NerResult(
            text=text,
            entities=self._parse_entities(str(result.final_output), original_text=text),
        )

    async def analyze_entities(
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
        verbose: bool = False,
    ) -> "NerResult":
        if str_or_none(text) is None:
            raise ValueError("text is required")

        chat_model = self._to_chat_model(model)

        class SimpleEntitiesResult(pydantic.BaseModel):
            entities: list[str] = pydantic.Field(default_factory=list)

        agent_instructions: str = (
            jinja2.Template(self.simple_entities_instructions)
            .render(fact_text=text)
            .strip()
        )

        if verbose:
            print("\n\n--- LLM INSTRUCTIONS ---\n")
            print(agent_instructions)

        agent = agents.Agent(
            name="simple-entities-agent",
            model=chat_model,
            model_settings=agents.ModelSettings(temperature=0.0),
            instructions=agent_instructions,
            output_type=SimpleEntitiesResult,
        )
        result = await agents.Runner.run(
            agent, text, run_config=agents.RunConfig(tracing_disabled=tracing_disabled)
        )

        if verbose:
            print("\n\n--- LLM OUTPUT ---\n")
            print(str(result.final_output))
            print("\n--- LLM USAGE ---\n")
            print(
                "Usage:",
                json.dumps(
                    asdict(result.context_wrapper.usage),
                    ensure_ascii=False,
                    default=str,
                ),
            )

        entities_result = result.final_output_as(SimpleEntitiesResult)

        entities: list[Entity] = []
        used_spans: list[tuple[int, int]] = []
        for entity in entities_result.entities:
            start_pos, end_pos = _claim_span(text, entity, used_spans)
            entities.append(
                Entity(name=entity, value=entity, start=start_pos, end=end_pos)
            )

        return NerResult(text=text, entities=entities)

    async def analyze_synonyms_and_canonical_name(
        self,
        candidate_list: list[str],
        *,
        model: (
            agents.OpenAIChatCompletionsModel
            | agents.OpenAIResponsesModel
            | ChatModel
            | str
            | None
        ) = None,
        tracing_disabled: bool = True,
        verbose: bool = False,
    ) -> "SynonymsAndCanonicalNameResult":
        if not candidate_list:
            raise ValueError("candidate_list is required")

        chat_model = self._to_chat_model(model)

        agent_instructions: str = (
            jinja2.Template(self.synonyms_and_canonical_name_instructions)
            .render(candidate_list=json.dumps(candidate_list, ensure_ascii=False))
            .strip()
        )

        if verbose:
            print("\n\n--- LLM INSTRUCTIONS ---\n")
            print(agent_instructions)

        agent = agents.Agent(
            name="synonyms-and-canonical-name-agent",
            model=chat_model,
            model_settings=agents.ModelSettings(temperature=0.0),
            instructions=agent_instructions,
            output_type=SynonymsAndCanonicalNameResult,
        )

        result = await agents.Runner.run(
            agent,
            agent_instructions,
            run_config=agents.RunConfig(tracing_disabled=tracing_disabled),
        )

        if verbose:
            print("\n\n--- LLM OUTPUT ---\n")
            print(str(result.final_output))
            print("\n--- LLM USAGE ---\n")
            print(
                "Usage:",
                json.dumps(
                    asdict(result.context_wrapper.usage),
                    ensure_ascii=False,
                    default=str,
                ),
            )

        return result.final_output_as(SynonymsAndCanonicalNameResult)

    def _parse_entities(
        self,
        entity_string: str,
        original_text: str = "",
    ) -> list["Entity"]:
        """
        Parse entities from strings containing zero or more occurrences of the pattern
        [ENTITY_TEXT](#ENTITY_TYPE). Robust to arbitrary whitespace, newlines, and
        missing pipe separators.

        Args:
            entity_string: Raw model output containing entity markup.
            original_text: Original source text (optional, but recommended for spans).

        Returns:
            List[Entity]
        """
        if not entity_string:
            return []

        # Global pattern: [text](#TYPE)
        pattern = re.compile(
            r"\[([^\]]+)\]\s*\(\s*#\s*([^)]+?)\s*\)", flags=re.IGNORECASE
        )

        entities: list[Entity] = []
        used_spans: list[tuple[int, int]] = []

        for m in pattern.finditer(entity_string):
            entity_text = m.group(1).strip()
            raw_type = m.group(2).strip().upper()

            ent_type = legacy_entity_map.get(raw_type, raw_type)

            # Skip unknown types to avoid validation errors downstream.
            if ent_type not in EntityType.__members__:
                if ent_type != "DONE":
                    logger.warning(f"Unknown entity type: {raw_type}")
                continue

            start_pos, end_pos = _claim_span(original_text, entity_text, used_spans)

            entities.append(
                Entity(
                    name=ent_type,
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


Entities = pydantic.TypeAdapter(list[Entity])


class NerResult(pydantic.BaseModel):
    text: str
    entities: list[Entity] = pydantic.Field(default_factory=list)


class SynonymsAndCanonicalNameResult(pydantic.BaseModel):
    """Pydantic model for parsing the synonyms and canonical name agent's output."""

    is_synonymous: bool
    canonical_name: str | None = None


def _claim_span(
    original_text: str, surface: str, used_spans: list[tuple[int, int]]
) -> tuple[int, int]:
    """
    Find the next non-overlapping occurrence of `surface` in original_text.
    Returns (-1, -1) if not found or no original_text was supplied.
    """
    if not original_text:
        return (0, 0)  # maintain current default behavior when text unknown

    # Search all occurrences; pick the first that doesn't overlap a prior claim.
    for mt in re.finditer(re.escape(surface), original_text):
        s, e = mt.span()
        if all(not (s < ue and e > us) for us, ue in used_spans):
            used_spans.append((s, e))
            return (s, e)
    return (-1, -1)
