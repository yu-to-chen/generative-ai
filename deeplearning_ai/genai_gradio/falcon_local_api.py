import os, traceback, torch
from fastapi import FastAPI, Response
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------- Config (fast & stable on M-series) --------
MODEL_ID = os.getenv("FALCON_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE    = torch.float32  # safest on Apple silicon

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
torch.set_default_dtype(torch.float32)

# -------- Load once --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# Align pad->eos if missing; left padding for causal models
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,                  # NOTE: new arg name (no deprecation)
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(DEVICE).eval()

# Ensure config has pad/eos IDs
if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
    model.config.pad_token_id = tokenizer.pad_token_id
if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
    model.config.eos_token_id = tokenizer.eos_token_id

# Text-generation pipeline
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

def build_prompt(user_text: str) -> str:
    """
    Format for chat models if a chat template exists.
    Falls back to raw prompt for base models.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_text

class GenRequest(BaseModel):
    inputs: str
    parameters: dict | None = None

app = FastAPI()

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
    }

@app.post("/generate")
def generate(req: GenRequest):
    try:
        p = req.parameters or {}
        max_new_tokens       = int(p.get("max_new_tokens", 64))
        min_new_tokens       = int(p.get("min_new_tokens", 8))   # ensure we get something
        temperature          = float(p.get("temperature", 0.7))
        top_p                = float(p.get("top_p", 0.9))
        do_sample            = bool(p.get("do_sample", True))
        repetition_penalty   = float(p.get("repetition_penalty", 1.0))
        no_repeat_ngram_size = int(p.get("no_repeat_ngram_size", 0))

        prompt_text = build_prompt(req.inputs)

        res = gen(
            prompt_text,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,   # returns just the completion if supported
        )

        text = res[0].get("generated_text", "")
        if not text:
            # Fallback: some versions ignore return_full_text; trim manually
            full = res[0]["generated_text"]
            text = full[len(prompt_text):].strip() if full.startswith(prompt_text) else full

        # Last resort: ensure non-empty by nudging params
        if not text.strip():
            res2 = gen(
                prompt_text,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max(min_new_tokens, 4),
                do_sample=True,
                temperature=max(temperature, 0.8),
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
            )
            text = res2[0].get("generated_text", "").strip()

        return [{"generated_text": text}]
    except Exception as e:
        tb = traceback.format_exc(limit=6)
        return Response(
            content=f'{{"error":"{type(e).__name__}","detail":"{str(e)}","trace":"{tb}"}}',
            media_type="application/json",
            status_code=500,
        )

