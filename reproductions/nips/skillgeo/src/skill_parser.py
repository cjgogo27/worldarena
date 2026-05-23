import re

from dataclasses import asdict, dataclass

from typing import Any





@dataclass

class Skill:

    skill_text: str

    region_hint: str

    confidence: float

    visual_cues: list[str]

    source_game_id: str

    source_round: int



    def to_dict(self) -> dict[str, Any]:

        return asdict(self)





COUNTRY_TO_REGION = {

    "al": "europe", "ad": "europe", "at": "europe", "by": "europe", "be": "europe", "ba": "europe", "bg": "europe",

    "hr": "europe", "cy": "europe", "cz": "europe", "dk": "europe", "ee": "europe", "fi": "europe", "fr": "europe",

    "de": "europe", "gr": "europe", "hu": "europe", "is": "europe", "ie": "europe", "it": "europe", "xk": "europe",

    "lv": "europe", "li": "europe", "lt": "europe", "lu": "europe", "mt": "europe", "md": "europe", "mc": "europe",

    "me": "europe", "nl": "europe", "mk": "europe", "no": "europe", "pl": "europe", "pt": "europe", "ro": "europe",

    "ru": "europe", "sm": "europe", "rs": "europe", "sk": "europe", "si": "europe", "es": "europe", "se": "europe",

    "ch": "europe", "ua": "europe", "gb": "europe", "va": "europe", "tr": "asia", "am": "asia", "az": "asia",

    "bh": "asia", "bd": "asia", "bt": "asia", "bn": "asia", "kh": "asia", "cn": "asia", "ge": "asia", "hk": "asia",

    "in": "asia", "id": "asia", "ir": "asia", "iq": "asia", "il": "asia", "jp": "asia", "jo": "asia", "kz": "asia",

    "kw": "asia", "kg": "asia", "la": "asia", "lb": "asia", "mo": "asia", "my": "asia", "mv": "asia", "mn": "asia",

    "mm": "asia", "np": "asia", "kp": "asia", "om": "asia", "pk": "asia", "ph": "asia", "qa": "asia", "sa": "asia",

    "sg": "asia", "kr": "asia", "lk": "asia", "sy": "asia", "tw": "asia", "tj": "asia", "th": "asia", "tl": "asia",

    "tm": "asia", "ae": "asia", "uz": "asia", "vn": "asia", "ye": "asia", "ca": "north_america", "us": "north_america",

    "mx": "north_america", "gt": "north_america", "bz": "north_america", "sv": "north_america", "hn": "north_america",

    "ni": "north_america", "cr": "north_america", "pa": "north_america", "ag": "north_america", "bs": "north_america",

    "bb": "north_america", "cu": "north_america", "dm": "north_america", "do": "north_america", "gd": "north_america",

    "ht": "north_america", "jm": "north_america", "kn": "north_america", "lc": "north_america", "vc": "north_america",

    "tt": "north_america", "ar": "south_america", "bo": "south_america", "br": "south_america", "cl": "south_america",

    "co": "south_america", "ec": "south_america", "gy": "south_america", "py": "south_america", "pe": "south_america",

    "sr": "south_america", "uy": "south_america", "ve": "south_america", "dz": "africa", "ao": "africa", "bj": "africa",

    "bw": "africa", "bf": "africa", "bi": "africa", "cm": "africa", "cv": "africa", "cf": "africa", "td": "africa",

    "km": "africa", "cd": "africa", "dj": "africa", "eg": "africa", "gq": "africa", "er": "africa", "sz": "africa",

    "et": "africa", "ga": "africa", "gm": "africa", "gh": "africa", "gn": "africa", "gw": "africa", "ci": "africa",

    "ke": "africa", "ls": "africa", "lr": "africa", "ly": "africa", "mg": "africa", "mw": "africa", "ml": "africa",

    "mr": "africa", "mu": "africa", "yt": "africa", "ma": "africa", "mz": "africa", "na": "africa", "ne": "africa",

    "ng": "africa", "cg": "africa", "re": "africa", "rw": "africa", "sh": "africa", "st": "africa", "sn": "africa",

    "sc": "africa", "sl": "africa", "so": "africa", "za": "africa", "ss": "africa", "sd": "africa", "tz": "africa",

    "tg": "africa", "tn": "africa", "ug": "africa", "zm": "africa", "zw": "africa", "au": "oceania", "nz": "oceania",

    "fj": "oceania", "pg": "oceania", "ws": "oceania", "to": "oceania", "vu": "oceania", "sb": "oceania",

}





COUNTRY_NAME_TO_ISO2 = {

    "albania": "al", "andorra": "ad", "argentina": "ar", "armenia": "am", "australia": "au", "austria": "at",

    "azerbaijan": "az", "bahrain": "bh", "bangladesh": "bd", "belarus": "by", "belgium": "be", "bhutan": "bt",

    "bolivia": "bo", "bosnia": "ba", "bosnia and herzegovina": "ba", "botswana": "bw", "brazil": "br",

    "brunei": "bn", "bulgaria": "bg", "cambodia": "kh", "cameroon": "cm", "canada": "ca", "chile": "cl",

    "china": "cn", "colombia": "co", "congo": "cd", "costa rica": "cr", "croatia": "hr", "cuba": "cu",

    "cyprus": "cy", "czech republic": "cz", "czechia": "cz", "democratic republic of congo": "cd",

    "denmark": "dk", "dominican republic": "do", "drc": "cd", "ecuador": "ec", "egypt": "eg",

    "el salvador": "sv", "england": "gb", "estonia": "ee", "ethiopia": "et", "fiji": "fj", "finland": "fi",

    "france": "fr", "georgia": "ge", "germany": "de", "ghana": "gh", "greece": "gr", "guatemala": "gt",

    "honduras": "hn", "hong kong": "hk", "hungary": "hu", "iceland": "is", "india": "in", "indonesia": "id",

    "iran": "ir", "iraq": "iq", "ireland": "ie", "israel": "il", "italy": "it", "jamaica": "jm",

    "japan": "jp", "jordan": "jo", "kazakhstan": "kz", "kenya": "ke", "kuwait": "kw", "kyrgyzstan": "kg",

    "laos": "la", "latvia": "lv", "lebanon": "lb", "libya": "ly", "liechtenstein": "li", "lithuania": "lt",

    "luxembourg": "lu", "macau": "mo", "madagascar": "mg", "malawi": "mw", "malaysia": "my", "mali": "ml",

    "malta": "mt", "mexico": "mx", "moldova": "md", "monaco": "mc", "mongolia": "mn", "montenegro": "me",

    "morocco": "ma", "mozambique": "mz", "myanmar": "mm", "namibia": "na", "nepal": "np",

    "netherlands": "nl", "new zealand": "nz", "nicaragua": "ni", "niger": "ne", "nigeria": "ng",

    "north korea": "kp", "north macedonia": "mk", "norway": "no", "oman": "om", "pakistan": "pk",

    "panama": "pa", "papua new guinea": "pg", "paraguay": "py", "peru": "pe", "philippines": "ph",

    "poland": "pl", "portugal": "pt", "qatar": "qa", "romania": "ro", "russia": "ru", "rwanda": "rw",

    "saudi arabia": "sa", "scotland": "gb", "senegal": "sn", "serbia": "rs", "singapore": "sg",

    "slovakia": "sk", "slovenia": "si", "somalia": "so", "south africa": "za", "south korea": "kr",

    "spain": "es", "sri lanka": "lk", "sudan": "sd", "sweden": "se", "switzerland": "ch",

    "syria": "sy", "taiwan": "tw", "tajikistan": "tj", "tanzania": "tz", "thailand": "th",

    "tunisia": "tn", "turkey": "tr", "turkmenistan": "tm", "uganda": "ug", "ukraine": "ua",

    "united arab emirates": "ae", "uae": "ae", "united kingdom": "gb", "uk": "gb",

    "united states": "us", "usa": "us", "uruguay": "uy", "uzbekistan": "uz", "venezuela": "ve",

    "vietnam": "vn", "wales": "gb", "yemen": "ye", "zambia": "zm", "zimbabwe": "zw",

}





VISUAL_CUE_KEYWORDS = {

    "road markings", "lane markings", "bollard", "sign", "license plate", "architecture", "vegetation", "soil", "mountains",

    "sun position", "driving", "pole", "poles", "road", "trees", "roof", "chevron", "crosswalk", "building", "car", "script",

    "cyrillic", "arabic", "fence", "bridge", "highway", "sidewalk", "guardrail", "terrain", "climate", "ocean", "coast",

}





def infer_region_from_text(text: str) -> str:

    lowered = text.lower()

    for country_name, iso in COUNTRY_NAME_TO_ISO2.items():

        if country_name in lowered:

            return COUNTRY_TO_REGION.get(iso, "unknown")

    return "unknown"





def infer_country_iso_from_text(text: str) -> str | None:

    lowered = text.lower()

    for country_name, iso in sorted(COUNTRY_NAME_TO_ISO2.items(), key=lambda kv: -len(kv[0])):

        if country_name in lowered:

            return iso

    return None





def extract_visual_cues(skill_text: str) -> list[str]:

    lowered = skill_text.lower()

    cues = [k for k in VISUAL_CUE_KEYWORDS if k in lowered]

    if not cues:

        token_candidates = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", skill_text)

        cues = token_candidates[:4]

    return sorted(set(cues))





def estimate_confidence(skill_text: str) -> float:

    lowered = skill_text.lower()

    if any(w in lowered for w in ["100%", "definitely", "highly likely", "certain"]):

        return 0.9

    if any(w in lowered for w in ["likely", "probably", "most likely"]):

        return 0.75

    if any(w in lowered for w in ["could", "maybe", "possibly", "ambiguous", "uncertain"]):

        return 0.55

    return 0.65





def parse_candidate_chain(text: str, source_game_id: str, round_num: int) -> list[Skill]:

    """Parse a candidate_reasoning_chain_gpt4_{N}.txt file.

    Format: plain list of observation lines, no Round/Conclusion headers.
    Last non-empty line(s) typically contain the conclusion country/city.
    Each non-empty line becomes a Skill; region_hint is inferred from the
    conclusion line (last line that contains a known country name).
    """

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if not lines:

        return []







    conclusion_text = ""

    for ln in reversed(lines):

        if infer_region_from_text(ln) != "unknown":

            conclusion_text = ln

            break

    if not conclusion_text:

        conclusion_text = lines[-1]



    region_hint = infer_region_from_text(conclusion_text)



    skills: list[Skill] = []

    for ln in lines:



        if len(ln) < 15:

            continue

        skills.append(

            Skill(

                skill_text=ln,

                region_hint=region_hint,

                confidence=estimate_confidence(ln),

                visual_cues=extract_visual_cues(ln),

                source_game_id=source_game_id,

                source_round=round_num,

            )

        )

    return skills





def parse_expert_chain(expert_text: str, source_game_id: str) -> list[Skill]:

    lines = [ln.strip() for ln in expert_text.splitlines()]

    skills: list[Skill] = []



    current_round = None

    current_conclusion = ""

    reasoning_lines: list[str] = []

    in_conclusion = False



    def flush_round() -> None:

        nonlocal reasoning_lines, current_conclusion

        if current_round is None:

            return

        region_hint = infer_region_from_text(current_conclusion)

        for rl in reasoning_lines:

            if not rl:

                continue

            if rl.lower().startswith("round ") or rl.lower() == "reasoning":

                continue

            skills.append(

                Skill(

                    skill_text=rl,

                    region_hint=region_hint,

                    confidence=estimate_confidence(rl),

                    visual_cues=extract_visual_cues(rl),

                    source_game_id=source_game_id,

                    source_round=current_round,

                )

            )

        reasoning_lines = []

        current_conclusion = ""



    for line in lines:

        if not line:

            continue

        m = re.match(r"^Round\s+(\d+)$", line, re.IGNORECASE)

        if m:

            flush_round()

            current_round = int(m.group(1))

            in_conclusion = False

            continue

        if line.lower() == "reasoning":

            in_conclusion = False

            continue

        if line.lower() == "conclusion":

            in_conclusion = True

            continue

        if in_conclusion:

            current_conclusion = (current_conclusion + " " + line).strip()

        else:

            reasoning_lines.append(line)



    flush_round()

    return skills
