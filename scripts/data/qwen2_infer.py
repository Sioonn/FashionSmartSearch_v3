"""
Qwen2.5-14B-Instruct simple inference script with selectable backends:
- "fp16" (default): standard fp16/bf16 load
- "qlora": 4-bit NF4 (bitsandbytes)
- "awq": AWQ quantized checkpoint (AutoAWQ)

Edit the variables in the USER SETTINGS section, then run:
  python scripts/data/qwen2_infer.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ===== USER SETTINGS =====
# Backend: "fp16" | "qlora" | "awq"
backend: str = "qlora"

# Model (for awq, use a pre-quantized repo/folder)
model_name: str = "Qwen/Qwen2.5-14B-Instruct"

# Prompt content
query: str = "Rewrite the following FashionGen-style technical description into the structured captions as per system rules: [description] Relaxed fit lounge pants in deep navy blue. Green honeycomb effect throughout weave. Three-pocket styling. Drawstring at waistband. Logo patch at ankle cuff. Tonal stitching. Short sleeve crewneck t-shirt in black. Signature red, white, and navy trim at chest pocket. Buttoned side-seam vents with signature trim. Tonal stitching. Long sleeve piqu&eacute; hoodie in dark green. Navy rib knit trim and panelling throughout. Drawstring closure at hood. Two-way zip closure and welt pockets, and logo patch at front. Signature tri-color trim at back collar and hem. Tonal stitching."
system_prompt: Optional[str] = """
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

3. No Hallucination

Use only information stated in the input.
You may paraphrase (e.g., “piqué” → “니트 재질”) but never invent new details.

4. Allowed Korean Fashion Terms

Use only these as-is:
트러커 재킷, MA-1, 점퍼, 코치 자켓, 야상, 파카, 라이더 자켓, 무스탕 재킷, 바람막이, 아노락, 맨투맨, 후드티, 후드 집업, 가디건, 니트, 스웨터, 하프집업, 헨리넥, 폴라넥, 터틀넥, 모크넥, 드롭 숄더, 투웨이 지퍼, 스트랩, 쭈리, 기모, 시보리, 플리스, 오버핏, 세미오버, 레귤러핏, 크롭, 박시, 슬림, 절개, 스티치, 누빔

All other expert terms → replace with simple visual phrasing (e.g., “technical satin” → “약간 광택 있는 소재”).

5. Korean Sentence Style

Natural descriptive sentences (1–2 sentences).

Avoid “보여줘 / 비슷한 거 / 찾아줘”.

Use tones like “~인데”, “~느낌의”, “~가 특징이다”.

Match the human query style examples:

“어두운 버건디색 맨투맨인데 아이보리 색 글씨가 전면에 적혀 있다.”

“베이지색 더블코트에 안감은 체크무늬가 있다.”

“검은색 데님 자켓에 얇은 스티치 포인트가 있다.”

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
  "지퍼 부분에 영어가 세로로 적혀 있고 검은색 모자를 쓴 공 캐릭터가 가슴쪽에 작게 있는 검은색 자켓",
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
"""
history: Optional[List[Dict[str, str]]] = None


def _bf16_supported() -> bool:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (8, 0)
    return False


def load_model_and_tokenizer(backend: str, model_id: str):
    if backend == "qlora":
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if _bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        return mdl, tok

    if backend == "awq":
        from awq import AutoAWQForCausalLM  # type: ignore

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        mdl = AutoAWQForCausalLM.from_quantized(
            model_id,
            fuse_layers=True,
            trust_remote_code=True,
            device_map="auto",
            safetensors=True,
        )
        return mdl, tok

    # default fp16/bf16 path
    dtype = torch.bfloat16 if _bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    return mdl, tok


def main() -> None:
    assert isinstance(query, str) and query.strip(), "Set a non-empty 'query' string."

    model, tokenizer = load_model_and_tokenizer(backend, model_name)

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    if torch.cuda.is_available():
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    with torch.no_grad():
        gen_out = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
        )

    # Remove the prompt tokens
    trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs["input_ids"], gen_out)
    ]
    response = tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
    print(response)


if __name__ == "__main__":
    main()

