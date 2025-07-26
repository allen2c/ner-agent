# ner-agent

A simple, language-agnostic Named Entity Recognition (NER) agent powered by LLMs.

## Features

- __Multilingual__: Supports English, Chinese, Japanese, Korean, French, German, Russian, Spanish, and more.
- __Entity Types__: Recognizes PERSON, NORP (nationalities, religions, political groups, languages), LOCATION, DATETIME, NUMERIC, and PROPER_NOUN (events, works, organizations, products, etc.).
- __Easy Integration__: Use as a Python library with a simple async API.
- __Tested__: Includes comprehensive test cases for multiple languages and entity types.

## Installation

```bash
pip install ner-agent
```

Or clone and install locally:

```bash
git clone https://github.com/allen2c/ner-agent.git
cd ner-agent
pip install .
```

__Python 3.11+ required.__

## Usage

```python
import asyncio
from ner_agent import NerAgent

async def main():
    agent = NerAgent()
    text = "Elon Musk visited Tesla's Gigafactory in Austin on March 15, 2024, announcing a 20% increase."
    result = await agent.run(text)
    for entity in result.entities:
        print(entity)

asyncio.run(main())
```

__Output:__

```plaintext
name='PERSON' value='Elon Musk' start=0 end=9
name='PROPER_NOUN' value='Tesla' start=18 end=23
name='LOCATION' value='Gigafactory' start=26 end=37
...
```

## Entity Types

- `PERSON`: People, including fictional characters.
- `NORP`: Nationalities, religious groups, political groups, languages.
- `LOCATION`: Geopolitical entities, facilities, places.
- `DATETIME`: Dates, times, periods, ages.
- `NUMERIC`: Numbers, money, quantities, percentages, ordinals/cardinals.
- `PROPER_NOUN`: Named events, works, laws, products, organizations, companies, etc.

## Testing

To run the tests:

```bash
pytest
```

## Configuration

- By default, uses OpenAI-compatible LLMs via [openai-agents](https://pypi.org/project/openai-agents/).
- You can configure the model and OpenAI client (see `tests/conftest.py` for examples).

## License

MIT License

---

For more details, see [ner_agent/__init__.py](ner_agent/__init__.py) and the [tests](tests/test_ner_agent_run.py).

If you need more advanced usage or want to contribute, please check the [GitHub repository](https://github.com/allen2c/ner-agent).
