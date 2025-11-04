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
Your goal is to read multiple short English technical description sentences (FashionGen-style) and rewrite only the parts related to tops or outerwear into natural, human-like descriptive captions in both English and Korean.
Each caption must sound exactly like what a real user would type when describing or searching for a clothing item online.

🎨 Input

You will receive multiple short English technical sentences describing several fashion items.
Each sentence provides visual details such as material, color, pattern, trim, logo, or shape.
For example:

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

🔍 Task Rules

1️⃣ Category Selection

Generate outputs only for categories:

Outerwear (outer) → e.g., jacket, coat, cardigan, blazer

Top (top) → e.g., t-shirt, shirt, blouse, hoodie, sweatshirt, knitwear

2️⃣ Output Format
For each valid item, generate:

Outer (Jacket)
영어: "Indigo denim jacket featuring an all-over orange embroidered pattern and an open front."
한국어: "전체적으로 오렌지색 자수 패턴이 들어간 인디고 데님 자켓이고, 앞부분이 트여있는 디자인이다."

Top (T-shirt)
영어: "White short-sleeve cotton t-shirt with a rib-knit crewneck and a signature stitched accent at the back hem."
한국어: "흰색 반팔 코튼 티셔츠인데 목은 시보리 크루넥이고, 뒤쪽 밑단에 스티치 포인트가 있다."


3️⃣ Language Style

The English output should be fluent, concise, and natural — like a product search query or descriptive caption.

The Korean output should sound natural and human-like — as if a user is describing a product they saw online.

Avoid unnatural literal translations or dictionary-like tone.

4️⃣ Natural Language Emphasis (Important)

⚠️ Do not use difficult or highly technical fashion terms from the input directly in the output.
You must rewrite them into natural, easy-to-understand language.
For example, “rib knit crewneck collar” should be rewritten as “ribbed crewneck,” and “tonal stitching” should be described naturally as “matching-color stitching” or “tone-on-tone stitching.”
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

