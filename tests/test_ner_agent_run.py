# tests/test_ner_agent_run.py
import typing

import agents
import pytest

from ner_agent import Entity, EntityType, NerAgent

TEST_CASES: list[tuple[str, str, list[Entity]]] = [
    (
        "id_en_elon",
        "Elon Musk visited Tesla's Gigafactory in Austin on March 15, 2024, announcing a 20% increase.",  # noqa: E501
        [
            Entity(name=EntityType.PERSON, value="Elon Musk", start=0, end=9),
            Entity(name=EntityType.ORG, value="Tesla", start=18, end=23),
            Entity(name=EntityType.FAC, value="Gigafactory", start=26, end=37),
            Entity(name=EntityType.LOCATION, value="Austin", start=41, end=47),
            Entity(name=EntityType.DATETIME, value="March 15, 2024", start=51, end=65),
            Entity(name=EntityType.NUMERIC, value="20%", start=80, end=83),
        ],
    ),
    (
        "id_en_law",
        "The Peace Treaty was signed on July 4, 2020 and extended last year.",
        [
            Entity(name=EntityType.LAW, value="Peace Treaty", start=4, end=16),
            Entity(name=EntityType.DATETIME, value="July 4, 2020", start=31, end=43),
            Entity(name=EntityType.DATETIME, value="last year", start=57, end=66),
        ],
    ),
    (
        "id_en_amazon",
        "Amazon sold 1,000 Echo Dots in Q4 2023 for $150,000.",
        [
            Entity(name=EntityType.ORG, value="Amazon", start=0, end=6),
            Entity(name=EntityType.NUMERIC, value="1,000", start=12, end=17),
            Entity(name=EntityType.PRODUCT, value="Echo Dots", start=18, end=27),
            Entity(name=EntityType.DATETIME, value="Q4 2023", start=31, end=38),
            Entity(name=EntityType.NUMERIC, value="$150,000", start=43, end=51),
        ],
    ),
    (
        "id_en_jfk",
        "Flights were diverted to JFK Airport after storms hit New Jersey on September 9, 2022.",  # noqa: E501
        [
            Entity(name=EntityType.FAC, value="JFK Airport", start=25, end=36),
            Entity(name=EntityType.LOCATION, value="New Jersey", start=54, end=64),
            Entity(
                name=EntityType.DATETIME, value="September 9, 2022", start=68, end=85
            ),
        ],
    ),
    (
        "id_en_lotr",
        "'The Lord of the Rings' was published in 1954 in London.",
        [
            Entity(
                name=EntityType.WORK_OF_ART,
                value="The Lord of the Rings",
                start=1,
                end=22,
            ),
            Entity(name=EntityType.DATETIME, value="1954", start=41, end=45),
            Entity(name=EntityType.LOCATION, value="London", start=49, end=55),
        ],
    ),
    (
        "id_en_katrina",
        "Hurricane Katrina devastated New Orleans on August 29, 2005.",
        [
            Entity(name=EntityType.EVENT, value="Hurricane Katrina", start=0, end=17),
            Entity(name=EntityType.LOCATION, value="New Orleans", start=29, end=40),
            Entity(name=EntityType.DATETIME, value="August 29, 2005", start=44, end=59),
        ],
    ),
    (
        "id_zh_apple",
        "蘋果公司在台北101發表了iPhone 15，預計售價為新台幣35,000元。",
        [
            Entity(name=EntityType.ORG, value="蘋果公司", start=0, end=4),
            Entity(name=EntityType.FAC, value="台北101", start=5, end=10),
            Entity(name=EntityType.PRODUCT, value="iPhone 15", start=13, end=22),
            Entity(name=EntityType.NUMERIC, value="新台幣35,000元", start=28, end=38),
        ],
    ),
    (
        "id_zh_law",
        "立法院於2023年12月25日通過了《個人資料保護法》修正案。",
        [
            Entity(name=EntityType.ORG, value="立法院", start=0, end=3),
            Entity(name=EntityType.DATETIME, value="2023年12月25日", start=4, end=15),
            Entity(name=EntityType.LAW, value="個人資料保護法", start=19, end=26),
        ],
    ),
    (
        "id_zh_tsmc",
        "昨天上午10點，台積電與台北市政府在台北簽署合作備忘錄。",
        [
            Entity(name=EntityType.DATETIME, value="昨天", start=0, end=2),
            Entity(name=EntityType.DATETIME, value="上午10點", start=2, end=7),
            Entity(name=EntityType.ORG, value="台積電", start=8, end=11),
            Entity(name=EntityType.ORG, value="台北市政府", start=12, end=17),
            Entity(name=EntityType.LOCATION, value="台北", start=18, end=20),
        ],
    ),
    (
        "id_zh_art",
        "日本人藝術家在國父紀念館展出《向日葵》系列作品。",
        [
            Entity(name=EntityType.NORP, value="日本人", start=0, end=3),
            Entity(name=EntityType.FAC, value="國父紀念館", start=7, end=12),
            Entity(name=EntityType.WORK_OF_ART, value="向日葵", start=15, end=18),
        ],
    ),
    (
        "id_ja_olympic",
        "東京オリンピックで日本人選手が金メダルを獲得した。",
        [
            Entity(name=EntityType.EVENT, value="東京オリンピック", start=0, end=8),
            Entity(name=EntityType.NORP, value="日本人", start=9, end=12),
            Entity(name=EntityType.PRODUCT, value="金メダル", start=15, end=19),
        ],
    ),
    (
        "id_ja_sony",
        "ソニーは2024年3月1日にPlayStation 6を発表した。",
        [
            Entity(name=EntityType.ORG, value="ソニー", start=0, end=3),
            Entity(name=EntityType.DATETIME, value="2024年3月1日", start=4, end=13),
            Entity(name=EntityType.PRODUCT, value="PlayStation 6", start=14, end=27),
        ],
    ),
    (
        "id_ja_kyoto",
        "京都駅は5月5日に大規模な改修工事を開始した。",
        [
            Entity(name=EntityType.FAC, value="京都駅", start=0, end=3),
            Entity(name=EntityType.DATETIME, value="5月5日", start=4, end=8),
        ],
    ),
    (
        "id_ko_samsung",
        "삼성전자는 서울 강남구에서 오전 9시에 갤럭시 S24를 공개했다.",
        [
            Entity(name=EntityType.ORG, value="삼성전자", start=0, end=4),
            Entity(name=EntityType.LOCATION, value="서울", start=6, end=8),
            Entity(name=EntityType.LOCATION, value="강남구", start=9, end=12),
            Entity(name=EntityType.DATETIME, value="오전 9시", start=15, end=20),
            Entity(name=EntityType.PRODUCT, value="갤럭시 S24", start=22, end=29),
        ],
    ),
    (
        "id_ko_law",
        "국회는 2024년 2월 10일에 데이터 보호법을 통과시키고 15%의 벌금을 부과했다.",
        [
            Entity(name=EntityType.ORG, value="국회", start=0, end=2),
            Entity(name=EntityType.DATETIME, value="2024년 2월 10일", start=4, end=16),
            Entity(name=EntityType.LAW, value="데이터 보호법", start=18, end=25),
            Entity(name=EntityType.NUMERIC, value="15%", start=33, end=36),
        ],
    ),
    (
        "id_ko_biff",
        "부산국제영화제는 2023년 10월 4일 부산에서 개막했다.",
        [
            Entity(name=EntityType.EVENT, value="부산국제영화제", start=0, end=7),
            Entity(name=EntityType.DATETIME, value="2023년 10월 4일", start=9, end=21),
            Entity(name=EntityType.LOCATION, value="부산", start=22, end=24),
        ],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("id,text,entities", TEST_CASES)
async def test_ner_agent_run(
    id: str,
    text: str,
    entities: typing.List[Entity],
    chat_model: agents.OpenAIChatCompletionsModel,
):
    agent = NerAgent()
    result = await agent.run(text, model=chat_model, verbose=True)

    assert set(e.name for e in result.entities) >= set(e.name for e in entities)
