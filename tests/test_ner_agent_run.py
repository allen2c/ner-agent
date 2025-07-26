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
            Entity(name=EntityType.LOCATION, value="Gigafactory", start=26, end=37),
            Entity(name=EntityType.LOCATION, value="Austin", start=41, end=47),
            Entity(name=EntityType.DATETIME, value="March 15, 2024", start=51, end=65),
            Entity(name=EntityType.NUMERIC, value="20%", start=80, end=83),
        ],
    ),
    (
        "id_en_law",
        "The Peace Treaty was signed on July 4, 2020 and extended last year.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Peace Treaty", start=4, end=16),
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
            Entity(name=EntityType.PROPER_NOUN, value="Echo Dots", start=18, end=27),
            Entity(name=EntityType.DATETIME, value="Q4 2023", start=31, end=38),
            Entity(name=EntityType.NUMERIC, value="$150,000", start=43, end=51),
        ],
    ),
    (
        "id_en_jfk",
        "Flights were diverted to JFK Airport after storms hit New Jersey on September 9, 2022.",  # noqa: E501
        [
            Entity(name=EntityType.LOCATION, value="JFK Airport", start=25, end=36),
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
                name=EntityType.PROPER_NOUN,
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
            Entity(
                name=EntityType.PROPER_NOUN, value="Hurricane Katrina", start=0, end=17
            ),
            Entity(name=EntityType.LOCATION, value="New Orleans", start=29, end=40),
            Entity(name=EntityType.DATETIME, value="August 29, 2005", start=44, end=59),
        ],
    ),
    (
        "id_zh_apple",
        "蘋果公司在台北101發表了iPhone 15，預計售價為新台幣35,000元。",
        [
            Entity(name=EntityType.ORG, value="蘋果公司", start=0, end=4),
            Entity(name=EntityType.LOCATION, value="台北101", start=5, end=10),
            Entity(name=EntityType.PROPER_NOUN, value="iPhone 15", start=13, end=22),
            Entity(name=EntityType.NUMERIC, value="新台幣35,000元", start=28, end=38),
        ],
    ),
    (
        "id_zh_law",
        "立法院於2023年12月25日通過了《個人資料保護法》修正案。",
        [
            Entity(name=EntityType.ORG, value="立法院", start=0, end=3),
            Entity(name=EntityType.DATETIME, value="2023年12月25日", start=4, end=15),
            Entity(
                name=EntityType.PROPER_NOUN, value="個人資料保護法", start=19, end=26
            ),
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
            Entity(name=EntityType.LOCATION, value="國父紀念館", start=7, end=12),
            Entity(name=EntityType.PROPER_NOUN, value="向日葵", start=15, end=18),
        ],
    ),
    (
        "id_ja_olympic",
        "東京オリンピックで日本人選手が金メダルを獲得した。",
        [
            Entity(
                name=EntityType.PROPER_NOUN, value="東京オリンピック", start=0, end=8
            ),
            Entity(name=EntityType.PROPER_NOUN, value="金メダル", start=15, end=19),
        ],
    ),
    (
        "id_ja_sony",
        "ソニーは2024年3月1日にPlayStation 6を発表した。",
        [
            Entity(name=EntityType.ORG, value="ソニー", start=0, end=3),
            Entity(name=EntityType.DATETIME, value="2024年3月1日", start=4, end=13),
            Entity(
                name=EntityType.PROPER_NOUN, value="PlayStation 6", start=14, end=27
            ),
        ],
    ),
    (
        "id_ja_kyoto",
        "京都駅は5月5日に大規模な改修工事を開始した。",
        [
            Entity(name=EntityType.LOCATION, value="京都駅", start=0, end=3),
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
            Entity(name=EntityType.PROPER_NOUN, value="갤럭시 S24", start=22, end=29),
        ],
    ),
    (
        "id_ko_law",
        "국회는 2024년 2월 10일에 데이터 보호법을 통과시키고 15%의 벌금을 부과했다.",
        [
            Entity(name=EntityType.ORG, value="국회", start=0, end=2),
            Entity(name=EntityType.DATETIME, value="2024년 2월 10일", start=4, end=16),
            Entity(
                name=EntityType.PROPER_NOUN, value="데이터 보호법", start=18, end=25
            ),
            Entity(name=EntityType.NUMERIC, value="15%", start=33, end=36),
        ],
    ),
    (
        "id_ko_biff",
        "부산국제영화제는 2023년 10월 4일 부산에서 개막했다.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="부산국제영화제", start=0, end=7),
            Entity(name=EntityType.DATETIME, value="2023년 10월 4일", start=9, end=21),
            Entity(name=EntityType.LOCATION, value="부산", start=22, end=24),
        ],
    ),
    (
        "id_en_openai",
        "OpenAI released GPT-5 on January 3, 2026, pricing the API at $0.002 per 1K tokens.",  # noqa: E501
        [
            Entity(name=EntityType.ORG, value="OpenAI"),
            Entity(name=EntityType.PROPER_NOUN, value="GPT-5"),
            Entity(name=EntityType.DATETIME, value="January 3, 2026"),
            Entity(name=EntityType.NUMERIC, value="$0.002"),
            Entity(name=EntityType.NUMERIC, value="1K"),
        ],
    ),
    # 2
    (
        "id_en_superbowl",
        "The Kansas City Chiefs won Super Bowl LVIII in Las Vegas on February 11, 2024, scoring 25 points.",  # noqa: E501
        [
            Entity(name=EntityType.ORG, value="Kansas City Chiefs"),
            Entity(name=EntityType.PROPER_NOUN, value="Super Bowl LVIII"),
            Entity(name=EntityType.LOCATION, value="Las Vegas"),
            Entity(name=EntityType.DATETIME, value="February 11, 2024"),
            Entity(name=EntityType.NUMERIC, value="25"),
        ],
    ),
    # 3
    (
        "id_en_gdpr",
        "Under the GDPR, Meta was fined €1.2 billion by the Irish DPC in May 2023.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="GDPR"),
            Entity(name=EntityType.ORG, value="Meta"),
            Entity(name=EntityType.NUMERIC, value="€1.2 billion"),
            Entity(name=EntityType.ORG, value="Irish DPC"),
            Entity(name=EntityType.DATETIME, value="May 2023"),
        ],
    ),
    # 4
    (
        "id_en_fraction",
        "The mixture contained 3/4 cup of sugar and 0.5 liters of milk on Monday morning.",  # noqa: E501
        [
            Entity(name=EntityType.NUMERIC, value="3/4"),
            Entity(name=EntityType.NUMERIC, value="0.5"),
            Entity(name=EntityType.DATETIME, value="Monday morning"),
        ],
    ),
    # 5
    (
        "id_en_quote_punct",
        '"Project Starlink" by SpaceX expanded to 60 countries as of last quarter.',
        [
            Entity(name=EntityType.ORG, value="SpaceX"),
            Entity(name=EntityType.NUMERIC, value="60"),
            Entity(name=EntityType.DATETIME, value="last quarter"),
        ],
    ),
    # 6
    (
        "id_en_overlap_repeat",
        "Apple opened a store in Apple Valley on 5/5/2025, and Apple fans lined up for 3 hours.",  # noqa: E501
        [
            Entity(name=EntityType.ORG, value="Apple"),
            Entity(name=EntityType.LOCATION, value="Apple Valley"),
            Entity(name=EntityType.DATETIME, value="5/5/2025"),
            Entity(name=EntityType.NUMERIC, value="3"),
        ],
    ),
    # 7
    (
        "id_en_range",
        "Between 2019 and 2021, Netflix grew its subscriber base by 30%.",
        [
            Entity(name=EntityType.DATETIME, value="2019"),
            Entity(name=EntityType.DATETIME, value="2021"),
            Entity(name=EntityType.ORG, value="Netflix"),
            Entity(name=EntityType.NUMERIC, value="30%"),
        ],
    ),
    # 8
    (
        "id_en_timezone",
        "The meeting starts at 14:30 UTC+2 on 7 September 2025 at the Eiffel Tower.",
        [
            Entity(name=EntityType.DATETIME, value="14:30 UTC+2"),
            Entity(name=EntityType.DATETIME, value="7 September 2025"),
            Entity(name=EntityType.LOCATION, value="Eiffel Tower"),
        ],
    ),
    # 9
    (
        "id_en_percent_ordinal",
        "Only the 3rd candidate received 12.5% of the votes in Q1 FY2024.",
        [
            Entity(name=EntityType.NUMERIC, value="3rd"),
            Entity(name=EntityType.NUMERIC, value="12.5%"),
            Entity(name=EntityType.DATETIME, value="Q1 FY2024"),
        ],
    ),
    # 10
    (
        "id_en_multiline",
        """NASA launched Artemis II\nfrom Kennedy Space Center on November 16, 2024.""",
        [
            Entity(name=EntityType.ORG, value="NASA"),
            Entity(name=EntityType.LOCATION, value="Kennedy Space Center"),
            Entity(name=EntityType.DATETIME, value="November 16, 2024"),
        ],
    ),
    # 11
    (
        "id_en_law_long",
        "The Clean Air Act amendments of 1990 were referenced alongside Title IX in Congress yesterday.",  # noqa: E501
        [
            Entity(name=EntityType.PROPER_NOUN, value="Clean Air Act"),
            Entity(name=EntityType.DATETIME, value="1990"),
            Entity(name=EntityType.PROPER_NOUN, value="Title IX"),
            Entity(name=EntityType.DATETIME, value="yesterday"),
        ],
    ),
    # 12
    (
        "id_en_currency_symbols",
        "They raised ¥500,000 in Tokyo on 12 Dec 2022 and another $3M in NYC.",
        [
            Entity(name=EntityType.NUMERIC, value="¥500,000"),
            Entity(name=EntityType.LOCATION, value="Tokyo"),
            Entity(name=EntityType.DATETIME, value="12 Dec 2022"),
            Entity(name=EntityType.NUMERIC, value="$3M"),
            Entity(name=EntityType.LOCATION, value="NYC"),
        ],
    ),
    # 13
    (
        "id_en_work_of_art_song",
        "Beyoncé performed 'Halo' at Madison Square Garden last night.",
        [
            Entity(name=EntityType.PERSON, value="Beyoncé"),
            Entity(name=EntityType.PROPER_NOUN, value="Halo"),
            Entity(name=EntityType.LOCATION, value="Madison Square Garden"),
            Entity(name=EntityType.DATETIME, value="last night"),
        ],
    ),
    # 14
    (
        "id_en_repeated_surface",
        "Paris officials met in Paris, Texas on June 6 to discuss tourism in Paris, France.",  # noqa: E501
        [
            Entity(name=EntityType.LOCATION, value="Paris"),
            Entity(name=EntityType.LOCATION, value="Paris, Texas"),
            Entity(name=EntityType.DATETIME, value="June 6"),
            Entity(name=EntityType.LOCATION, value="Paris, France"),
        ],
    ),
    # 15
    (
        "id_en_scientific",
        "CERN detected a particle at 13 TeV during Run 3 in 2022.",
        [
            Entity(name=EntityType.ORG, value="CERN"),
            Entity(name=EntityType.NUMERIC, value="13"),
            Entity(name=EntityType.PROPER_NOUN, value="Run 3"),
            Entity(name=EntityType.DATETIME, value="2022"),
        ],
    ),
    # 16
    (
        "id_en_email_phone",
        "Contact John Doe at john@example.com or call 555-1234 tomorrow at 9am.",
        [
            Entity(name=EntityType.PERSON, value="John Doe"),
            Entity(name=EntityType.DATETIME, value="tomorrow"),
            Entity(name=EntityType.DATETIME, value="9am"),
            Entity(name=EntityType.NUMERIC, value="555-1234"),
        ],
    ),
    # 17
    (
        "id_en_holiday",
        "Black Friday 2024 saw Apple sell 2 million iPhone 16 units.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Black Friday 2024"),
            Entity(name=EntityType.ORG, value="Apple"),
            Entity(name=EntityType.NUMERIC, value="2 million"),
            Entity(name=EntityType.PROPER_NOUN, value="iPhone 16"),
        ],
    ),
    # 18
    (
        "id_en_decade",
        "In the 1990s, Microsoft dominated the PC market.",
        [
            Entity(name=EntityType.DATETIME, value="1990s"),
            Entity(name=EntityType.ORG, value="Microsoft"),
            Entity(name=EntityType.PROPER_NOUN, value="PC"),
        ],
    ),
    # 19
    (
        "id_en_law_acronym",
        "Canada's C-11 bill amended the Broadcasting Act on April 27, 2023.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="C-11"),
            Entity(name=EntityType.PROPER_NOUN, value="Broadcasting Act"),
            Entity(name=EntityType.DATETIME, value="April 27, 2023"),
            Entity(name=EntityType.LOCATION, value="Canada"),
        ],
    ),
    # 20
    (
        "id_en_event_series",
        "COP28 in Dubai concluded on December 12, 2023 with a landmark agreement.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="COP28"),
            Entity(name=EntityType.LOCATION, value="Dubai"),
            Entity(name=EntityType.DATETIME, value="December 12, 2023"),
            Entity(name=EntityType.PROPER_NOUN, value="agreement"),
        ],
    ),
    # 21
    (
        "id_zh_money_range",
        "該公司在2024年第一季度收入介於新台幣1億到1.5億之間。",
        [
            Entity(name=EntityType.DATETIME, value="2024年第一季度"),
            Entity(name=EntityType.NUMERIC, value="新台幣1億"),
            Entity(name=EntityType.NUMERIC, value="1.5億"),
        ],
    ),
    # 22
    (
        "id_zh_event_relative",
        "上個月的臺北馬拉松吸引了3萬名跑者參加。",
        [
            Entity(name=EntityType.DATETIME, value="上個月"),
            Entity(name=EntityType.PROPER_NOUN, value="臺北馬拉松"),
            Entity(name=EntityType.NUMERIC, value="3萬"),
        ],
    ),
    # 23
    (
        "id_zh_law_gdpr",
        "根據《一般資料保護規則》（GDPR），公司必須在72小時內通報。",
        [
            Entity(name=EntityType.PROPER_NOUN, value="一般資料保護規則"),
            Entity(name=EntityType.PROPER_NOUN, value="GDPR"),
            Entity(name=EntityType.NUMERIC, value="72小時"),
        ],
    ),
    # 24
    (
        "id_zh_person_org",
        "馬英九昨日在中國國家博物館發表演說。",
        [
            Entity(name=EntityType.PERSON, value="馬英九"),
            Entity(name=EntityType.DATETIME, value="昨日"),
            Entity(name=EntityType.LOCATION, value="中國國家博物館"),
        ],
    ),
    # 25
    (
        "id_zh_percent_ordinal",
        "第三名的隊伍只拿到15%的得票率。",
        [
            Entity(name=EntityType.NUMERIC, value="第三"),
            Entity(name=EntityType.NUMERIC, value="15%"),
        ],
    ),
    # 26
    (
        "id_zh_duplicate_city",
        "上海市政府與上海交通大學於2022年6月1日簽署協議。",
        [
            Entity(name=EntityType.ORG, value="上海市政府"),
            Entity(name=EntityType.ORG, value="上海交通大學"),
            Entity(name=EntityType.DATETIME, value="2022年6月1日"),
            Entity(name=EntityType.LOCATION, value="上海"),
        ],
    ),
    # 27
    (
        "id_zh_work_of_art",
        "他在故宮博物院展出了《清明上河圖》真跡。",
        [
            Entity(name=EntityType.LOCATION, value="故宮博物院"),
            Entity(name=EntityType.PROPER_NOUN, value="清明上河圖"),
        ],
    ),
    # 28
    (
        "id_zh_time_relative",
        "我們明天下午三點在台中見面。",
        [
            Entity(name=EntityType.DATETIME, value="明天下午三點"),
            Entity(name=EntityType.LOCATION, value="台中"),
        ],
    ),
    # 29
    (
        "id_zh_event_disaster",
        "921大地震發生於1999年9月21日，造成嚴重傷亡。",
        [
            Entity(name=EntityType.PROPER_NOUN, value="921大地震"),
            Entity(name=EntityType.DATETIME, value="1999年9月21日"),
            Entity(name=EntityType.NUMERIC, value="嚴重"),
        ],
    ),
    # 30
    (
        "id_zh_norp_language",
        "台灣人常用中文與臺語交流。",
        [
            Entity(name=EntityType.NORP, value="台灣人"),
            Entity(name=EntityType.NORP, value="中文"),
            Entity(name=EntityType.NORP, value="臺語"),
        ],
    ),
    # 31
    (
        "id_ja_relative_week",
        "来週の水曜日、東京都庁で会議があります。",
        [
            Entity(name=EntityType.DATETIME, value="来週の水曜日"),
            Entity(name=EntityType.LOCATION, value="東京都庁"),
        ],
    ),
    # 32
    (
        "id_ja_money_percent",
        "任天堂は2025年に売上を20%増加させ、3兆円を達成した。",
        [
            Entity(name=EntityType.ORG, value="任天堂"),
            Entity(name=EntityType.DATETIME, value="2025年"),
            Entity(name=EntityType.NUMERIC, value="20%"),
            Entity(name=EntityType.NUMERIC, value="3兆円"),
        ],
    ),
    # 33
    (
        "id_ja_event_culture",
        "祇園祭は7月に京都で開催される有名な祭りです。",
        [
            Entity(name=EntityType.PROPER_NOUN, value="祇園祭"),
            Entity(name=EntityType.DATETIME, value="7月"),
            Entity(name=EntityType.LOCATION, value="京都"),
        ],
    ),
    # 34
    (
        "id_ja_law",
        "個人情報保護法に基づき、企業はデータを適切に管理しなければならない。",
        [
            Entity(name=EntityType.PROPER_NOUN, value="個人情報保護法"),
        ],
    ),
    # 35
    (
        "id_ja_facility_train",
        "新大阪駅で5時30分の新幹線に乗った。",
        [
            Entity(name=EntityType.LOCATION, value="新大阪駅"),
            Entity(name=EntityType.DATETIME, value="5時30分"),
        ],
    ),
    # 36
    (
        "id_ja_work_of_art",
        "『君の名は。』は2016年に公開された映画です。",
        [
            Entity(name=EntityType.PROPER_NOUN, value="君の名は。"),
            Entity(name=EntityType.DATETIME, value="2016年"),
        ],
    ),
    # 37
    (
        "id_ja_product_series",
        "ソニーはPlayStation VR2を2023年2月22日に発売した。",
        [
            Entity(name=EntityType.ORG, value="ソニー"),
            Entity(name=EntityType.PROPER_NOUN, value="PlayStation VR2"),
            Entity(name=EntityType.DATETIME, value="2023年2月22日"),
        ],
    ),
    # 38
    (
        "id_ko_relative",
        "다음 주 월요일 오전 10시에 서울시청에서 발표가 있다.",
        [
            Entity(name=EntityType.DATETIME, value="다음 주 월요일 오전 10시"),
            Entity(name=EntityType.LOCATION, value="서울시청"),
        ],
    ),
    # 39
    (
        "id_ko_event_sport",
        "월드컵 결승전은 2022년 12월 18일 카타르에서 열렸다.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="월드컵 결승전"),
            Entity(name=EntityType.DATETIME, value="2022년 12월 18일"),
            Entity(name=EntityType.LOCATION, value="카타르"),
        ],
    ),
    # 40
    (
        "id_ko_law_percent",
        "공정거래법 개정안은 2023년 국회를 통과하며 과징금을 5%로 상향했다.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="공정거래법"),
            Entity(name=EntityType.DATETIME, value="2023년"),
            Entity(name=EntityType.NUMERIC, value="5%"),
        ],
    ),
    # 41
    (
        "id_ko_facility_bridge",
        "한강대교는 1917년에 개통되었다.",
        [
            Entity(name=EntityType.LOCATION, value="한강대교"),
            Entity(name=EntityType.DATETIME, value="1917년"),
        ],
    ),
    # 42
    (
        "id_es_politics",
        "Pedro Sánchez anunció el plan en Madrid el 2 de mayo de 2023.",
        [
            Entity(name=EntityType.PERSON, value="Pedro Sánchez"),
            Entity(name=EntityType.LOCATION, value="Madrid"),
            Entity(name=EntityType.DATETIME, value="2 de mayo de 2023"),
        ],
    ),
    # 43
    (
        "id_es_event_music",
        "El festival Viña del Mar 2024 reunió a 50.000 espectadores.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Viña del Mar 2024"),
            Entity(name=EntityType.NUMERIC, value="50.000"),
        ],
    ),
    # 44
    (
        "id_es_law",
        "Según la Ley Orgánica de Protección de Datos, la empresa debe responder en 30 días.",  # noqa: E501
        [
            Entity(
                name=EntityType.PROPER_NOUN, value="Ley Orgánica de Protección de Datos"
            ),
            Entity(name=EntityType.NUMERIC, value="30"),
        ],
    ),
    # 45
    (
        "id_es_currency",
        "Barcelona pagó €750.000 por el jugador el año pasado.",
        [
            Entity(name=EntityType.ORG, value="Barcelona"),
            Entity(name=EntityType.NUMERIC, value="€750.000"),
            Entity(name=EntityType.DATETIME, value="el año pasado"),
        ],
    ),
    # 46
    (
        "id_es_work_of_art",
        "'Cien años de soledad' fue publicada en 1967 en Buenos Aires.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Cien años de soledad"),
            Entity(name=EntityType.DATETIME, value="1967"),
            Entity(name=EntityType.LOCATION, value="Buenos Aires"),
        ],
    ),
    # 47
    (
        "id_fr_event_sport",
        "Les Jeux Olympiques de Paris 2024 commenceront le 26 juillet 2024.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Jeux Olympiques de Paris 2024"),
            Entity(name=EntityType.DATETIME, value="26 juillet 2024"),
        ],
    ),
    # 48
    (
        "id_fr_law_gdpr",
        "Conformément au RGPD, l'entreprise doit notifier sous 72 heures.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="RGPD"),
            Entity(name=EntityType.NUMERIC, value="72"),
        ],
    ),
    # 49
    (
        "id_fr_fac_bridge",
        "Le Pont Neuf a été inauguré en 1607 à Paris.",
        [
            Entity(name=EntityType.LOCATION, value="Pont Neuf"),
            Entity(name=EntityType.DATETIME, value="1607"),
            Entity(name=EntityType.LOCATION, value="Paris"),
        ],
    ),
    # 50
    (
        "id_fr_product",
        "Renault a présenté la Megane E-Tech en 2022.",
        [
            Entity(name=EntityType.ORG, value="Renault"),
            Entity(name=EntityType.PROPER_NOUN, value="Megane E-Tech"),
            Entity(name=EntityType.DATETIME, value="2022"),
        ],
    ),
    # 51
    (
        "id_de_law",
        "Das Bundesdatenschutzgesetz (BDSG) wurde 2018 reformiert.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Bundesdatenschutzgesetz"),
            Entity(name=EntityType.PROPER_NOUN, value="BDSG"),
            Entity(name=EntityType.DATETIME, value="2018"),
        ],
    ),
    # 52
    (
        "id_de_event_fair",
        "Die Frankfurter Buchmesse 2023 zog 180.000 Besucher an.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Frankfurter Buchmesse 2023"),
            Entity(name=EntityType.NUMERIC, value="180.000"),
        ],
    ),
    # 53
    (
        "id_de_org_auto",
        "BMW investierte 1,2 Milliarden € in München im Jahr 2024.",
        [
            Entity(name=EntityType.ORG, value="BMW"),
            Entity(name=EntityType.NUMERIC, value="1,2 Milliarden €"),
            Entity(name=EntityType.LOCATION, value="München"),
            Entity(name=EntityType.DATETIME, value="2024"),
        ],
    ),
    # 54
    (
        "id_de_work_of_art",
        "'Die Verwandlung' wurde 1915 veröffentlicht.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Die Verwandlung"),
            Entity(name=EntityType.DATETIME, value="1915"),
        ],
    ),
    # 55
    (
        "id_ru_event_politics",
        "Выборы президента прошли 18 марта 2018 года в России.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Выборы президента"),
            Entity(name=EntityType.DATETIME, value="18 марта 2018 года"),
            Entity(name=EntityType.LOCATION, value="Россия"),
        ],
    ),
    # 56
    (
        "id_ru_law",
        "Согласно Федеральному закону №152-ФЗ о персональных данных, компании обязаны защищать информацию.",  # noqa: E501
        [
            Entity(name=EntityType.PROPER_NOUN, value="Федеральный закон №152-ФЗ"),
            Entity(name=EntityType.PROPER_NOUN, value="персональных данных"),
        ],
    ),
    # 57
    (
        "id_ru_product",
        "Яндекс представил сервис Яндекс.Плюс в 2018 году.",
        [
            Entity(name=EntityType.ORG, value="Яндекс"),
            Entity(name=EntityType.PROPER_NOUN, value="Яндекс.Плюс"),
            Entity(name=EntityType.DATETIME, value="2018 году"),
        ],
    ),
    # 58
    (
        "id_ar_event_religious",
        "انطلقت شعائر الحج في مكة في 14 يونيو 2024.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="شعائر الحج"),
            Entity(name=EntityType.LOCATION, value="مكة"),
            Entity(name=EntityType.DATETIME, value="14 يونيو 2024"),
        ],
    ),
    # 59
    (
        "id_ar_law",
        "وفقًا لقانون الجرائم الإلكترونية لعام 2015، تُفرض غرامة 10٪.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="قانون الجرائم الإلكترونية"),
            Entity(name=EntityType.DATETIME, value="2015"),
            Entity(name=EntityType.NUMERIC, value="10٪"),
        ],
    ),
    # 60
    (
        "id_ar_org_product",
        "شركة سامسونج أطلقت Galaxy Fold في 2019.",
        [
            Entity(name=EntityType.ORG, value="سامسونج"),
            Entity(name=EntityType.PROPER_NOUN, value="Galaxy Fold"),
            Entity(name=EntityType.DATETIME, value="2019"),
        ],
    ),
    # 61
    (
        "id_hi_event_festival",
        "कुंभ मेला 2021 में हरिद्वार में आयोजित किया गया।",
        [
            Entity(name=EntityType.PROPER_NOUN, value="कुंभ मेला"),
            Entity(name=EntityType.DATETIME, value="2021"),
            Entity(name=EntityType.LOCATION, value="हरिद्वार"),
        ],
    ),
    # 62
    (
        "id_hi_law",
        "सूचना का अधिकार अधिनियम 2005 के तहत नागरिक जानकारी मांग सकते हैं।",
        [
            Entity(name=EntityType.PROPER_NOUN, value="सूचना का अधिकार अधिनियम"),
            Entity(name=EntityType.DATETIME, value="2005"),
        ],
    ),
    # 63
    (
        "id_hi_org_product",
        "टाटा मोटर्स ने 2024 में टियागो ईवी लॉन्च की।",
        [
            Entity(name=EntityType.ORG, value="टाटा मोटर्स"),
            Entity(name=EntityType.DATETIME, value="2024"),
            Entity(name=EntityType.PROPER_NOUN, value="टियागो ईवी"),
        ],
    ),
    # 64
    (
        "id_en_emoji",
        "Tesla 🚗 delivered 400k cars in Q2 2023; Elon Musk tweeted 😀.",
        [
            Entity(name=EntityType.ORG, value="Tesla"),
            Entity(name=EntityType.NUMERIC, value="400k"),
            Entity(name=EntityType.DATETIME, value="Q2 2023"),
            Entity(name=EntityType.PERSON, value="Elon Musk"),
        ],
    ),
    # 65
    (
        "id_en_no_entities",
        "Just a plain sentence without any named things.",
        [],
    ),
    # 66
    (
        "id_en_relative_series",
        "Next month, IBM will host Think 2025 in Orlando.",
        [
            Entity(name=EntityType.DATETIME, value="Next month"),
            Entity(name=EntityType.ORG, value="IBM"),
            Entity(name=EntityType.PROPER_NOUN, value="Think 2025"),
            Entity(name=EntityType.LOCATION, value="Orlando"),
        ],
    ),
    # 67
    (
        "id_en_law_short",
        "Under HIPAA, hospitals must protect patient data.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="HIPAA"),
            Entity(name=EntityType.LOCATION, value="hospitals"),
        ],
    ),
    # 68
    (
        "id_en_event_war",
        "World War II ended in 1945 after Germany surrendered.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="World War II"),
            Entity(name=EntityType.DATETIME, value="1945"),
            Entity(name=EntityType.LOCATION, value="Germany"),
        ],
    ),
    # 69
    (
        "id_en_product_vehicle",
        "Ford unveiled the F-150 Lightning in May 2021.",
        [
            Entity(name=EntityType.ORG, value="Ford"),
            Entity(name=EntityType.PROPER_NOUN, value="F-150 Lightning"),
            Entity(name=EntityType.DATETIME, value="May 2021"),
        ],
    ),
    # 70
    (
        "id_en_fac_univ",
        "Classes at Harvard University resume on September 4, 2025.",
        [
            Entity(name=EntityType.ORG, value="Harvard University"),
            Entity(name=EntityType.DATETIME, value="September 4, 2025"),
        ],
    ),
    # 71
    (
        "id_en_numeric_big",
        "The galaxy is 2.5 million light-years away.",
        [
            Entity(name=EntityType.NUMERIC, value="2.5 million"),
        ],
    ),
    # 72
    (
        "id_en_law_constitution",
        "The U.S. Constitution was signed in 1787 in Philadelphia.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="U.S. Constitution"),
            Entity(name=EntityType.DATETIME, value="1787"),
            Entity(name=EntityType.LOCATION, value="Philadelphia"),
        ],
    ),
    # 73
    (
        "id_en_event_conference",
        "DEF CON 32 will take place in Las Vegas in August 2024.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="DEF CON 32"),
            Entity(name=EntityType.LOCATION, value="Las Vegas"),
            Entity(name=EntityType.DATETIME, value="August 2024"),
        ],
    ),
    # 74
    (
        "id_en_product_food",
        "Nestlé introduced KitKat Ruby in Japan in 2018.",
        [
            Entity(name=EntityType.ORG, value="Nestlé"),
            Entity(name=EntityType.PROPER_NOUN, value="KitKat Ruby"),
            Entity(name=EntityType.LOCATION, value="Japan"),
            Entity(name=EntityType.DATETIME, value="2018"),
        ],
    ),
    # 75
    (
        "id_en_event_hurricane",
        "Hurricane Ida hit Louisiana on August 29, 2021.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Hurricane Ida"),
            Entity(name=EntityType.LOCATION, value="Louisiana"),
            Entity(name=EntityType.DATETIME, value="August 29, 2021"),
        ],
    ),
    # 76
    (
        "id_en_fac_bridge",
        "The Golden Gate Bridge opened in 1937.",
        [
            Entity(name=EntityType.LOCATION, value="Golden Gate Bridge"),
            Entity(name=EntityType.DATETIME, value="1937"),
        ],
    ),
    # 77
    (
        "id_en_work_of_art_game",
        "The Legend of Zelda: Tears of the Kingdom launched in May 2023.",
        [
            Entity(
                name=EntityType.PROPER_NOUN,
                value="The Legend of Zelda: Tears of the Kingdom",
            ),
            Entity(name=EntityType.DATETIME, value="May 2023"),
        ],
    ),
    # 78
    (
        "id_en_numeric_ord_card",
        "She ranked 1st in a field of 2,345 participants.",
        [
            Entity(name=EntityType.NUMERIC, value="1st"),
            Entity(name=EntityType.NUMERIC, value="2,345"),
        ],
    ),
    # 79
    (
        "id_en_datetime_range_words",
        "From June to September 2024, tourists flocked to Bali.",
        [
            Entity(name=EntityType.DATETIME, value="June"),
            Entity(name=EntityType.DATETIME, value="September 2024"),
            Entity(name=EntityType.LOCATION, value="Bali"),
        ],
    ),
    # 80
    (
        "id_en_product_version",
        "Adobe released Photoshop 2025 Beta on April 1, 2025.",
        [
            Entity(name=EntityType.ORG, value="Adobe"),
            Entity(name=EntityType.PROPER_NOUN, value="Photoshop 2025 Beta"),
            Entity(name=EntityType.DATETIME, value="April 1, 2025"),
        ],
    ),
    # 81
    (
        "id_en_fac_airport",
        "He landed at LAX at 6:45 PM yesterday.",
        [
            Entity(name=EntityType.LOCATION, value="LAX"),
            Entity(name=EntityType.DATETIME, value="6:45 PM"),
            Entity(name=EntityType.DATETIME, value="yesterday"),
        ],
    ),
    # 82
    (
        "id_en_event_expo",
        "Expo 2020 Dubai actually ran from October 1, 2021 to March 31, 2022.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Expo 2020 Dubai"),
            Entity(name=EntityType.DATETIME, value="October 1, 2021"),
            Entity(name=EntityType.DATETIME, value="March 31, 2022"),
        ],
    ),
    # 83
    (
        "id_en_law_state",
        "California Consumer Privacy Act (CCPA) took effect on January 1, 2020.",
        [
            Entity(
                name=EntityType.PROPER_NOUN, value="California Consumer Privacy Act"
            ),
            Entity(name=EntityType.PROPER_NOUN, value="CCPA"),
            Entity(name=EntityType.DATETIME, value="January 1, 2020"),
            Entity(name=EntityType.LOCATION, value="California"),
        ],
    ),
    # 84
    (
        "id_en_event_film",
        "The Cannes Film Festival 2023 awarded the Palme d'Or to 'Anatomy of a Fall'.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Cannes Film Festival 2023"),
            Entity(name=EntityType.PROPER_NOUN, value="Anatomy of a Fall"),
        ],
    ),
    # 85
    (
        "id_en_product_ai",
        "Google unveiled Gemini Ultra on December 12, 2023.",
        [
            Entity(name=EntityType.ORG, value="Google"),
            Entity(name=EntityType.PROPER_NOUN, value="Gemini Ultra"),
            Entity(name=EntityType.DATETIME, value="December 12, 2023"),
        ],
    ),
    # 86
    (
        "id_en_event_music_award",
        "At the Grammys 2024, Taylor Swift won Album of the Year.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Grammys 2024"),
            Entity(name=EntityType.PERSON, value="Taylor Swift"),
        ],
    ),
    # 87
    (
        "id_en_fac_hospital",
        "St. Mary's Hospital admitted 120 patients on March 3.",
        [
            Entity(name=EntityType.LOCATION, value="St. Mary's Hospital"),
            Entity(name=EntityType.NUMERIC, value="120"),
            Entity(name=EntityType.DATETIME, value="March 3"),
        ],
    ),
    # 88
    (
        "id_en_event_science",
        "The Nobel Prize in Physics 2022 was awarded to Alain Aspect.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Nobel Prize in Physics 2022"),
            Entity(name=EntityType.PERSON, value="Alain Aspect"),
        ],
    ),
    # 89
    (
        "id_en_numeric_temp",
        "Temperatures hit 42°C in Delhi last week.",
        [
            Entity(name=EntityType.NUMERIC, value="42°C"),
            Entity(name=EntityType.LOCATION, value="Delhi"),
            Entity(name=EntityType.DATETIME, value="last week"),
        ],
    ),
    # 90
    (
        "id_en_event_marathon",
        "The Boston Marathon 2023 saw 30,000 runners finish.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Boston Marathon 2023"),
            Entity(name=EntityType.NUMERIC, value="30,000"),
        ],
    ),
    # 91
    (
        "id_en_work_of_art_album",
        "'OK Computer' by Radiohead was released in 1997.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="OK Computer"),
            Entity(name=EntityType.ORG, value="Radiohead"),
            Entity(name=EntityType.DATETIME, value="1997"),
        ],
    ),
    # 92
    (
        "id_en_event_space",
        "SpaceX's Starship reached orbit for the first time on November 18, 2023.",
        [
            Entity(name=EntityType.ORG, value="SpaceX"),
            Entity(name=EntityType.PROPER_NOUN, value="Starship"),
            Entity(name=EntityType.DATETIME, value="November 18, 2023"),
        ],
    ),
    # 93
    (
        "id_en_fac_museum",
        "The Louvre Museum reported 8.7 million visitors in 2022.",
        [
            Entity(name=EntityType.LOCATION, value="Louvre Museum"),
            Entity(name=EntityType.NUMERIC, value="8.7 million"),
            Entity(name=EntityType.DATETIME, value="2022"),
        ],
    ),
    # 94
    (
        "id_en_org_merger",
        "AT&T merged with Time Warner in 2018 for $85 billion.",
        [
            Entity(name=EntityType.ORG, value="AT&T"),
            Entity(name=EntityType.ORG, value="Time Warner"),
            Entity(name=EntityType.DATETIME, value="2018"),
            Entity(name=EntityType.NUMERIC, value="$85 billion"),
        ],
    ),
    # 95
    (
        "id_en_datetime_iso",
        "The log was created at 2025-07-26T09:15:00Z.",
        [
            Entity(name=EntityType.DATETIME, value="2025-07-26T09:15:00Z"),
        ],
    ),
    # 96
    (
        "id_en_event_award_film",
        "Academy Awards 2020 honored 'Parasite' as Best Picture.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Academy Awards 2020"),
            Entity(name=EntityType.PROPER_NOUN, value="Parasite"),
        ],
    ),
    # 97
    (
        "id_en_law_eu",
        "The Digital Markets Act entered into force on November 1, 2022.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="Digital Markets Act"),
            Entity(name=EntityType.DATETIME, value="November 1, 2022"),
        ],
    ),
    # 98
    (
        "id_en_fac_highway",
        "Interstate 95 was closed for 12 hours after the accident.",
        [
            Entity(name=EntityType.LOCATION, value="Interstate 95"),
            Entity(name=EntityType.NUMERIC, value="12"),
        ],
    ),
    # 99
    (
        "id_en_product_food2",
        "Coca-Cola Zero Sugar was relaunched in 2021.",
        [
            Entity(name=EntityType.ORG, value="Coca-Cola"),
            Entity(name=EntityType.PROPER_NOUN, value="Zero Sugar"),
            Entity(name=EntityType.DATETIME, value="2021"),
        ],
    ),
    # 100
    (
        "id_en_event_fair2",
        "CES 2025 will open in Las Vegas on January 7, 2025.",
        [
            Entity(name=EntityType.PROPER_NOUN, value="CES 2025"),
            Entity(name=EntityType.LOCATION, value="Las Vegas"),
            Entity(name=EntityType.DATETIME, value="January 7, 2025"),
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
