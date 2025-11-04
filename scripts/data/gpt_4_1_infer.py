import os
import json
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


def load_api_key() -> str:
    """Load OpenAI API key from environment supporting both names."""
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API key not found. Set OPENAI_API_KEY or OPEN_AI_API_KEY in .env"
        )
    return api_key


def read_system_prompt() -> str:
    """Read system prompt from scripts/data/README.md to avoid duplication."""
    readme_path = Path(__file__).with_name("README.md")
    with readme_path.open("r", encoding="utf-8") as f:
        return f.read()


def load_inputs(jsonl_path: Path, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
    """Load JSONL entries filtered by inclusive idx range."""
    items: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj.get("idx"), int) and start_idx <= obj["idx"] <= end_idx:
                items.append(obj)
    items.sort(key=lambda o: o.get("idx", 0))
    return items


def load_all_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load all rows from a JSONL file preserving order."""
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Preserve malformed lines as-is in a wrapper to avoid data loss
                rows.append({"_raw": line})
    return rows


def write_all_jsonl(jsonl_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows back to JSONL with one object per line."""
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if "_raw" in row and len(row) == 1:
                f.write(str(row["_raw"]))
            else:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    # Backup original once
    bak_path = jsonl_path.with_suffix(jsonl_path.suffix + ".bak")
    if not bak_path.exists():
        try:
            bak_path.write_text(jsonl_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    tmp_path.replace(jsonl_path)


def main() -> None:
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    model = "gpt-4.1-mini-2025-04-14"
    system_prompt = '''
        You are a Fashion Language Transformer.
        Your role is to read short English technical description sentences (FashionGen-style) and rewrite only the parts related to tops or outerwear into natural, human-like descriptive captions in both English and Korean.

        Each caption must sound exactly like what a real user would type or say when describing or searching for a clothing item online.

        🎨 Input

        You will receive multiple short English technical sentences describing several fashion items.
        Each sentence provides visual details such as material, color, pattern, trim, logo, or shape.

        Example input:

        Long sleeve cotton fleece hoodie in pale heather grey.  
        Tonal drawstring and cotton lining featuring signature check pattern in tan, black, red, and white at hood.  
        Zip closure at front.  
        Logo embroidered at chest.  
        Kangaroo pockets at waist.  
        Rib knit cuffs and hem.  
        Long sleeve cotton and wool-blend houndstooth blazer in tan, sea turtle green, and black.  
        Notched lapel collar.  
        Double-breasted button closure at front.  

        🔍 Task Rules
        1. Category Selection

        Generate output only for:

        top → shirt, t-shirt, sweatshirt, hoodie, knitwear, blouse, cardigan, vest, sleeveless top

        outer → jacket, coat, blazer, trench coat, leather jacket, parka, windbreaker
        Ignore pants, skirts, shorts, and accessories.

        2. Visible-Only Restriction

        Describe only what is visually observable – color, texture, pattern, logo/text, collar, hood, zipper, pockets, shape, silhouette, or stitch details.
        Do not mention hidden/internal details (lining, padding, fiber composition, insulation, inner seams).

        3. Expert Fashion Term Expansion (전문 용어 풀어서 쓰기) When encountering professional or technical fashion terms (e.g., lapel, satin trim, welt pocket, vent, surgeon’s cuff, grain de poudre, gabardine, piqué, crepe, twill, mock cuff, grosgrain, ruching, toggle, colorblocked, etc.), do not use the term directly. Instead, rewrite it as a natural, visual, and intuitive description that an ordinary person would use when describing what they see.
        For example:
        “lapel collar” → “pointed collar shape” / “카라 부분이 뾰족하게 되어 있음”
        “satin trim” → “slightly shiny edge detail” / “가장자리가 은은하게 반짝이는 테두리 디테일”
        “welt pocket” → “flat, neatly sewn pocket” / “얇고 깔끔하게 마감된 포켓”
        “vented cuffs” → “small slit at sleeve ends” / “소매 끝부분에 트임이 있음”
        “surgeon’s cuff” → “sleeve buttons near the wrist” / “손목 부분에 단추 장식이 있음”
        “gabardine” → “firm, smooth wool fabric” / “탄탄하고 매끈한 질감의 원단”
        “ruching” → “gathered folds or wrinkles” / “자연스럽게 주름이 잡힌 부분”
        “toggle fastening” → “looped button closure” / “고리형 단추 여밈”
        The goal is to describe what it looks or feels like, not to repeat technical terminology. Never add new details not stated in the input.


        4. Allowed Korean Fashion Terms

        Allow these as-is: 
        트러커 재킷, MA-1, 점퍼, 코치 자켓, 야상, 파카, 라이더 자켓, 무스탕 재킷, 바람막이, 아노락, 맨투맨, 후드티, 후드 집업, 가디건, 니트, 스웨터, 하프집업, 헨리넥, 폴라넥, 터틀넥, 모크넥, 드롭 숄더, 투웨이 지퍼, 스트랩, 쭈리, 기모, 시보리, 플리스, 오버핏, 세미오버, 레귤러핏, 크롭, 박시, 슬림, 절개, 스티치, 누빔, 카라 

        All other expert terms → replace with simple visual phrasing (e.g., “technical satin” → “약간 광택 있는 소재”).

        5. Korean Sentence Style

        Natural descriptive sentences (1–2 sentences).

        Avoid “보여줘 / 비슷한 거 / 찾아줘”.

        Use tones like “~인데”, “~느낌의”, “~가 특징이다”.

        Match the human query style examples:

        “어두운 버건디색 맨투맨인데 아이보리 색 글씨가 전면에 적혀 있다.”

        “베이지색 더블코트에 안감은 체크무늬가 있다.”

        “검은색 데님 자켓에 얇은 스티치 포인트가 있다.”
        6. Diversity Rule (문장 다양성 규칙) Each Korean output must include 3 independent sentences that express the same meaning, but use different synonyms, tone, or sentence structure to feel natural and human-like.
        All three Korean captions must describe the same garment clearly, but vary in expression (e.g., word order, adjectives, phrasing).
        Use synonyms and different sentence endings such as “~인데”, “~느낌의”, “~가 특징이다”, “~으로 보인다”, “~같다”.
        Occasionally (about 10–20% of cases), you may include natural search-intent endings like “~찾아줘” or “~찾아” when it fits a realistic search query.
        Example (for a black denim jacket):
        “검은색 데님 재킷으로 전체적으로 주황색 자수가 들어가 있다.”
        “블랙 데님 자켓인데 곳곳에 오렌지색 자수 포인트가 있는 스타일이다.”
        “인디고 톤의 자수가 박힌 검은 데님 재킷 찾아줘.”
        All three share the same meaning but differ in tone and word choice.


        🔒 Independence & Diversity Rules

        Independence Rule
        Every English and Korean sentence must stand alone and fully describe the item (type + color + key features).
        ❌ No continuations like “has a pocket”.
        ✅ Use complete forms like “네이비색 코튼 재킷으로 앞면에 큼직한 단추 여밈이 있다.”

        Diversity Rule

        Vary word order and focus (“색 먼저 → 디테일”, “디테일 먼저 → 색”).

        For Korean: alternate patterns (“~인데”, “~로”, “~느낌의”).

        For English: vary between “A … with …”, “This … features …”, “Minimal … defined by …”.

        Redundancy Allowed
        Repeating item type or color is okay and encouraged for clarity.

        💬 Context Learning Examples (Use as Style Reference)

        These are “gold-standard” examples of human-like Korean descriptions.
        Your outputs should feel similarly natural and fluent.

        [
        "전체적으로 파란색에 흰색 포인트가 곳곳에 들어간 바시티 자켓",
        "지퍼 부분에 영어가 세로로 적혀 있고 검은색 모자를 쓴 공 캐릭터가 가슴쪽에 작게 있는 검은색 자켓을 찾아줘.", 
        "소매 부분에 흰색 검은색 네이비색 선이 길게 있고 옷 아래부분과 카라가 흰색이며, 전체적으로는 검은색인 자켓",
        "검은색 데님 자켓에 양쪽 가슴에 주머니가 있고, 곳곳에 얇은 선 형태의 스티치 포인트가 있음",
        "몸통 부분은 청색 데님인데, 소매와 후드 모자가 회색이고 후드에는 끈도 있음"
        ]

        🧾 Output Format

        Output only categories present in the input.
        Each category includes:

        English: 2 independent captions

        Korean: 3 independent captions

        Output must be valid JSON in this schema:

        {
        "top": {
            "english": ["<...>", "<...>"],
            "korean": ["<...>", "<...>", "<...>"]
        },
        "outer": {
            "english": ["<...>", "<...>"],
            "korean": ["<...>", "<...>", "<...>"]
        }
        }

        🧱 User Prompt Template
        Rewrite the following FashionGen-style technical description into the structured captions as per the system rules:

        <Long English concatenated description here>

        🧩 Example – Full Demonstration

        Input

        Low-top suede sneakers in 'sand' tan.  
        Beading in green and white, and moccasin-style stitching at round toe.  
        Tonal leather lace-up closure.  
        Signature stitching in beige at tongue.  
        Fringed suede overlay at collar.  
        Fringe at heel.  
        Rubber sole colorblocked in beige and black featuring logo at outer heel.  
        Tonal stitching and contrast stitching in brown.  
        Long sleeve denim jacket in indigo featuring embroidered pattern in orange throughout.  
        Open front.  
        Tonal stitching.  
        Wide-leg cotton canvas trousers in navy.  
        Purple tinge throughout.  
        High-rise.  
        Five-pocket styling.  
        Signature handstitched accent in white at back waistband.  
        Adjustable cinch tab at back yoke.  
        Button-fly.  
        Tonal stitching.  
        Short sleeve cotton jersey t-shirt in white.  
        Rib knit crewneck collar.  
        Signature handstitched accent at back hem.


        Output

        {
        "top": {
            "english": [
            "White short-sleeve cotton jersey t-shirt with a rib-knit crewneck and a signature handstitched accent at the back hem.",
            "Minimal white cotton tee featuring a clean crewneck design and subtle handstitched detail at the back."
            ],
            "korean": [
            "흰색 코튼 반팔 티셔츠로 시보리 크루넥과 뒤쪽 밑단의 핸드스티치 디테일이 있다.",
            "화이트 코튼 티셔츠인데 깔끔한 크루넥 디자인과 뒤쪽의 섬세한 스티치 포인트가 특징이다.",
            "심플한 흰색 반팔 티셔츠로 시그니처 핸드스티치가 뒷단에 작게 들어가 있다."
            ]
        },
        "outer": {
            "english": [
            "Indigo denim jacket with long sleeves and orange embroidered patterns throughout.",
            "Long-sleeve indigo denim jacket featuring allover orange embroidery and tonal stitching."
            ],
            "korean": [
            "인디고 데님 재킷으로 긴소매에 주황색 자수가 전체적으로 들어가 있다.",
            "짙은 인디고 색상의 데님 자켓으로 전체적으로 오렌지색 자수 패턴이 새겨져 있다.",
            "인디고 데님 재킷인데 곳곳에 주황색 자수 장식이 들어간 독특한 스타일이다."
            ]
        }
        }

        '''

    data_path = Path("data/preprocess/desc_raw/tops_concat_descriptions.jsonl")
    items = load_inputs(data_path, start_idx=0, end_idx=10)
    # Load entire JSONL to update in-place
    all_rows = load_all_jsonl(data_path)
    rows_by_idx: Dict[int, Dict[str, Any]] = {
        r["idx"]: r for r in all_rows if isinstance(r, dict) and "idx" in r
    }

    json_schema = {
        "type": "object",
        "properties": {
            "top": {
                "type": "object",
                "properties": {
                    "english": {"type": "array", "items": {"type": "string"}},
                    "korean": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["english", "korean"],
                "additionalProperties": False,
            },
            "outer": {
                "type": "object",
                "properties": {
                    "english": {"type": "array", "items": {"type": "string"}},
                    "korean": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["english", "korean"],
                "additionalProperties": False,
            },
        },
        # Responses API requires required to include every key in properties
        "required": ["top", "outer"],
        "additionalProperties": False,
    }

    for obj in items:
        idx = obj.get("idx")
        user_content = (obj.get("input_concat_description") or "").strip()
        if not user_content:
            print(f"[skip] idx={idx}: empty input_concat_description")
            continue

        print(f"[request] idx={idx} → model={model}")
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        + "\n\nSchema constraint: Always include both 'top' and 'outer' keys in the JSON output. If a category is not present in the input, return empty arrays for that category (e.g., 'english': [], 'korean': [])."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ffs_v3_fashion_caption",
                    "strict": True,
                    "schema": json_schema,
                }
            },
        )

        # Prefer convenience property; fall back to raw structure
        text = getattr(response, "output_text", None)
        if not text:
            try:
                text = response.output[0].content[0].text
            except Exception:
                text = ""

        preview = (text or "").replace("\n", " ")
        print(f"[response] idx={idx}: {preview[:160]}...\n")

        # Update preprocess field in corresponding JSONL row
        target = rows_by_idx.get(idx)
        if target is None:
            print(f"[warn] idx={idx} not found in JSONL; skipping write")
            continue
        try:
            parsed = json.loads(text)
            target["preprocess"] = parsed
            print(f"[updated] idx={idx} preprocess set (parsed JSON)")
        except Exception:
            target["preprocess"] = text or ""
            print(f"[updated] idx={idx} preprocess set (raw text)")

    # Persist all rows back to the JSONL
    write_all_jsonl(data_path, all_rows)
    print(f"[saved] Updated JSONL → {data_path}")


if __name__ == "__main__":
    main()
