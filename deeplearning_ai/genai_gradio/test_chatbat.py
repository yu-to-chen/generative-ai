# one-time (in your venv with uv):
# uv pip install "torch>=2.2" transformers accelerate

import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Apple Silicon friendly defaults ---
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE  = torch.float32  # safest/stable on M-series

# Small, fast local model (override via env if you want)
MODEL_ID = os.getenv("LOCAL_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# --- Load once (fast after first call) ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=DTYPE, trust_remote_code=True, low_cpu_mem_usage=True
).to(DEVICE).eval()

gen = pipeline("text-generation", model=model, tokenizer=tokenizer)  # model already on DEVICE

# --- Your function, upgraded to use chat templates (if present) ---
MAX_TURNS = 6  # keep short context for speed

def format_chat_prompt(message, chat_history, instruction):
    """
    Returns a single string prompt suitable for the local chat model.
    Uses tokenizer's chat template when available; otherwise falls back
    to your System/User/Assistant format.
    """
    # Trim history for speed
    trimmed = chat_history[-MAX_TURNS:] if chat_history else []

    # Prefer proper chat formatting (works best with *-Chat / *-Instruct models)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "system", "content": instruction or "You are a helpful assistant."}]
        for u, b in trimmed:
            if u: msgs.append({"role": "user", "content": u})
            if b: msgs.append({"role": "assistant", "content": b})
        msgs.append({"role": "user", "content": message})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Fallback to your original concatenation
    prompt = f"System:{instruction or 'You are a helpful assistant.'}"
    for u, b in trimmed:
        prompt = f"{prompt}\nUser: {u}\nAssistant: {b}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

# --- Local, fast responder using the formatted prompt ---
def respond(message, chat_history, instruction="", max_new_tokens=128):
    if not message or not str(message).strip():
        return "", chat_history
    prompt_text = format_chat_prompt(message, chat_history, instruction)

    out = gen(
        prompt_text,
        max_new_tokens=int(max_new_tokens),  # speed knob
        min_new_tokens=8,                    # avoid empty replies
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,              # returns just the completion if supported
    )

    reply = out[0].get("generated_text", "").strip()
    if not reply:  # fallback trim if older transformers ignores return_full_text
        full = out[0]["generated_text"]
        reply = full[len(prompt_text):].strip() if full.startswith(prompt_text) else full.strip()

    chat_history.append((message, reply))
    return "", chat_history

# --- Example (no UI) ---
if __name__ == "__main__":
    history = []
    _, history = respond("Hi! Who are you?", history, instruction="Be concise.")
    _, history = respond("Write a 1-sentence fun fact about greyhounds.", history, instruction="Be concise.")
    print(history[-1][1])

