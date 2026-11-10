import json
import math
import os
import re
from pathlib import Path

# Keep numerical libraries from starting many threads on small free instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("ORBIS_MODEL_DIR", BASE / "model"))
WEIGHTS_PATH = Path(os.getenv("ORBIS_WEIGHTS", MODEL_DIR / "orbis_step6_weights.npz"))
TOKENIZER_PATH = Path(os.getenv("ORBIS_TOKENIZER", MODEL_DIR / "tokenizer.json"))

class ByteBPETokenizer:
    def __init__(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "orbis-bpe-v1":
            raise RuntimeError("Unsupported Orbis tokenizer format")
        self.vocab_size = int(data["vocab_size"])
        self.merges = data["merges"]
        self.ranks, self.children = {}, {}
        for rank, item in enumerate(self.merges):
            left, right, new = int(item["left"]), int(item["right"]), int(item["new"])
            self.ranks[(left, right)] = rank
            self.children[new] = (left, right)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            best_rank, best_pair = None, None
            for a, b in zip(tokens, tokens[1:]):
                rank = self.ranks.get((a, b))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, (a, b)
            if best_pair is None:
                break
            a, b = best_pair
            new_id = int(self.merges[best_rank]["new"])
            merged, i = [], 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i+1] == b:
                    merged.append(new_id); i += 2
                else:
                    merged.append(tokens[i]); i += 1
            tokens = merged
        return tokens

    def decode(self, ids):
        output = bytearray()
        def emit(token_id):
            if 0 <= token_id < 256:
                output.append(token_id)
            elif token_id in self.children:
                left, right = self.children[token_id]
                emit(left); emit(right)
            else:
                output.extend(b"?")
        for token_id in ids:
            emit(int(token_id))
        return output.decode("utf-8", errors="replace")


def linear(x, prefix, weights):
    return x @ weights[prefix + ".weight"].T + weights[prefix + ".bias"]

def layer_norm(x, prefix, weights, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weights[prefix + ".weight"] + weights[prefix + ".bias"]

def gelu(x):
    # Tanh approximation is numerically close to PyTorch's exact GELU.
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

class OrbisNumpyModel:
    def __init__(self, weights_path, metadata_path):
        self.wfile = np.load(weights_path, allow_pickle=False)
        self.w = {k: self.wfile[k] for k in self.wfile.files}
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.vocab_size = int(meta["vocab_size"])
        self.block_size = int(meta["block_size"])
        self.d_model = int(meta["d_model"])
        self.n_heads = int(meta["n_heads"])
        self.n_layers = int(meta["n_layers"])
        self.head_dim = self.d_model // self.n_heads

    def __call__(self, ids):
        # ids is a 1-D list of token IDs. The network is the same Step 6 architecture.
        ids = ids[-self.block_size:]
        x = self.w["token.weight"][np.asarray(ids, dtype=np.int64)]
        x = x + self.w["pos.weight"][np.arange(len(ids), dtype=np.int64)]
        length = x.shape[0]
        for layer in range(self.n_layers):
            p = f"blocks.{layer}"
            h = layer_norm(x, p + ".ln1", self.w)
            qkv = linear(h, p + ".attn.qkv", self.w)
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(length, self.n_heads, self.head_dim).transpose(1, 0, 2)
            k = k.reshape(length, self.n_heads, self.head_dim).transpose(1, 0, 2)
            v = v.reshape(length, self.n_heads, self.head_dim).transpose(1, 0, 2)
            scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
            scores = np.where(np.triu(np.ones((length, length), dtype=bool), 1)[None, :, :], -1e9, scores)
            scores = scores - scores.max(axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn = attn / attn.sum(axis=-1, keepdims=True)
            z = (attn @ v).transpose(1, 0, 2).reshape(length, self.d_model)
            x = x + linear(z, p + ".attn.out", self.w)
            h = layer_norm(x, p + ".ln2", self.w)
            h = gelu(linear(h, p + ".ff.0", self.w))
            h = linear(h, p + ".ff.2", self.w)
            x = x + h
        x = layer_norm(x, "ln", self.w)
        return x[-1] @ self.w["head.weight"].T


# ---- Orbis v2: the model you trained yourself with train.py (put converted files in model2/) ----
ORBIS2_DIR = Path(os.getenv("ORBIS2_DIR", BASE / "model2"))
orbis2 = None
if (ORBIS2_DIR / "orbis_weights.npz").exists():
    try:
        from orbis_np import OrbisNP
        orbis2 = OrbisNP(ORBIS2_DIR)
        orbis2.tok   # load the tokenizer now so a problem shows up at startup, not on the first request
        print(f"[orbis] v2 model loaded: {orbis2.num_params:,} parameters")
    except Exception as exc:
        orbis2 = None
        print(f"[orbis] could not load model2/: {type(exc).__name__}: {exc}")

# ---- Orbis Step 6 (the original 387K model); optional once v2 is in place ----
try:
    tokenizer = ByteBPETokenizer(TOKENIZER_PATH)
    model = OrbisNumpyModel(WEIGHTS_PATH, MODEL_DIR / "model_metadata.json")
except Exception as exc:
    tokenizer = model = None
    if orbis2 is None:
        raise
LOCAL_LABEL = (f"Studio Orbis v2 ({orbis2.num_params:,} parameters)" if orbis2
               else "Studio Orbis Step 6 (386,688 parameters)")
LOCAL_PARAMS = orbis2.num_params if orbis2 else 386688
app = FastAPI(title="Studio Orbis AI", version="0.8.0-orbis-qwen-fallback")
origins = [x.strip() for x in os.getenv("ORBIS_ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=["*"] if origins == ["*"] else origins,
                   allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


# ---- Optional Gemma runtime (local folder or Hugging Face model repository) ----
# Keep this disabled until the large model weights and matching tokenizer are available.
GEMMA_MODEL = os.getenv("ORBIS_GEMMA_MODEL", "").strip()
GEMMA_TOKEN = os.getenv("HF_TOKEN", "").strip()
gemma_model = None
gemma_tokenizer = None
gemma_device = None
if GEMMA_MODEL:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        gemma_tokenizer = AutoTokenizer.from_pretrained(
            GEMMA_MODEL, token=GEMMA_TOKEN or None, trust_remote_code=False
        )
        # Let Transformers choose the checkpoint's stored dtype. CPU hosting still
        # needs several GB of RAM for a 1B model; this is not suitable for tiny Render instances.
        gemma_model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL, token=GEMMA_TOKEN or None, torch_dtype="auto",
            low_cpu_mem_usage=True, trust_remote_code=False
        )
        gemma_device = "cuda" if torch.cuda.is_available() else "cpu"
        gemma_model.to(gemma_device)
        gemma_model.eval()
        print(f"[orbis] Gemma model loaded from {GEMMA_MODEL} on {gemma_device}")
    except Exception as exc:
        gemma_model = gemma_tokenizer = None
        print(f"[orbis] Gemma model not loaded: {type(exc).__name__}: {exc}")


def gemma_generate(message, max_new_tokens=256, temperature=0.7):
    if gemma_model is None or gemma_tokenizer is None:
        raise RuntimeError("Gemma is not configured")
    import torch
    prompt = "You are Studio Orbis AI. Answer helpfully and clearly in English unless the user asks for another language.\n\nUser: " + message + "\nAssistant:"
    inputs = gemma_tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(gemma_device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = gemma_model.generate(
            **inputs, max_new_tokens=max(1, min(int(max_new_tokens), 1024)),
            do_sample=temperature > 0, temperature=max(0.1, float(temperature)),
            pad_token_id=gemma_tokenizer.eos_token_id
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return gemma_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def tiny_generate(message, max_new_tokens=48, temperature=0.8, top_k=30):
    """Draft text from the homemade model (v2 if loaded, otherwise Step 6)."""
    if orbis2 is not None:
        # Orbis v2 models trained with train.py --sft expect this chat template.
        return orbis2.complete(f"User: {message.strip()}\nAssistant:",
                               max_new_tokens, temperature, top_k)
    ids = tokenizer.encode("Question: " + message.strip() + "\nAnswer:")
    if not ids: ids = [32]
    generated = []
    for _ in range(max(1, min(max_new_tokens, 96))):
        logits = model(ids + generated) / temperature
        k = min(top_k, logits.shape[-1])
        top = np.argpartition(logits, -k)[-k:]
        vals = logits[top]
        vals = vals - vals.max()
        probs = np.exp(vals); probs /= probs.sum()
        generated.append(int(np.random.choice(top, p=probs)))
    return tokenizer.decode(generated).strip()


# ---- Optional Qwen GGUF fallback (served separately by llama.cpp) ----
# Render hosts Orbis. A separate machine must run llama.cpp and expose an
# OpenAI-compatible /v1 endpoint. Hugging Face is the model-file source only.
QWEN_FALLBACK_URL = os.getenv("ORBIS_QWEN_URL", os.getenv("QWEN_FALLBACK_URL", "")).strip().rstrip("/")
QWEN_FALLBACK_MODEL = os.getenv("ORBIS_QWEN_MODEL", "qwen3.8-flash-next").strip()
QWEN_FALLBACK_API_KEY = os.getenv("ORBIS_QWEN_API_KEY", "").strip()
QWEN_FALLBACK_TIMEOUT = float(os.getenv("ORBIS_QWEN_TIMEOUT", "180"))
QWEN_FALLBACK_ENABLED = os.getenv("ORBIS_QWEN_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def looks_unusable(text):
    """Detect obvious malformed/repetitive output; not a semantic fact checker."""
    if not isinstance(text, str) or not text.strip():
        return True, "empty_output"
    s = text.strip()
    if "\ufffd" in s:
        return True, "replacement_character"
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in s):
        return True, "control_characters"
    if len(s) >= 24:
        visible = sum(ch.isalnum() or ch.isspace() or ch in ".,!?;:'\"()[]{}<>+-=*/_`#@%&$\\\n\r" for ch in s)
        if visible / max(1, len(s)) < 0.68:
            return True, "mostly_nontext_symbols"
        if re.search(r"(.)\1{9,}", s):
            return True, "repeated_character_loop"
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if len(lines) >= 3 and max(lines.count(line) for line in set(lines)) >= 3:
        return True, "repeated_line_loop"
    words = re.findall(r"\w+", s.lower())
    if len(words) >= 18 and len(set(words)) / len(words) < 0.18:
        return True, "repetitive_word_loop"
    return False, "passed_basic_checks"


def qwen_fallback_chat(question, reason, max_tokens=768, temperature=0.7):
    """Ask a separately hosted llama.cpp-compatible Qwen server."""
    if not QWEN_FALLBACK_ENABLED or not QWEN_FALLBACK_URL:
        raise RuntimeError("Qwen fallback is not enabled or ORBIS_QWEN_URL is not configured")
    headers = {"Content-Type": "application/json"}
    if QWEN_FALLBACK_API_KEY:
        headers["Authorization"] = f"Bearer {QWEN_FALLBACK_API_KEY}"
    body = {
        "model": QWEN_FALLBACK_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are the backup model for Studio Orbis. The primary Orbis model "
                "could not produce a usable answer. Answer the user's original question "
                "clearly and accurately. Do not mention internal fallback routing."
            )},
            {"role": "user", "content": question},
        ],
        "max_tokens": max(1, min(int(max_tokens), 2048)),
        "temperature": max(0.0, min(float(temperature), 1.5)),
    }
    response = httpx.post(
        QWEN_FALLBACK_URL + "/chat/completions",
        json=body,
        headers=headers,
        timeout=QWEN_FALLBACK_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    answer = payload["choices"][0]["message"].get("content") or ""
    if isinstance(answer, list):
        answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
    answer = str(answer).strip()
    bad, why = looks_unusable(answer)
    if bad:
        raise RuntimeError(f"Qwen response failed output checks: {why}")
    return {
        "response": answer,
        "route": "qwen-fallback",
        "model": QWEN_FALLBACK_MODEL,
        "fallback_reason": reason,
        "primary_model": LOCAL_LABEL,
    }


def maybe_qwen_fallback(question, reason, request):
    if not QWEN_FALLBACK_ENABLED or not QWEN_FALLBACK_URL:
        return None
    return qwen_fallback_chat(
        question, reason,
        max_tokens=request.max_new_tokens or 768,
        temperature=request.temperature,
    )

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    top_k: int = Field(default=30, ge=1, le=100)
    debug: bool = False   # include routing info and the tiny-model draft in the reply
    force_qwen_fallback: bool = False  # manually test the separate Qwen backup

@ app.get("/")
def root():
    return {"name":"Studio Orbis AI", "status":"online", "step":10,
            "model_parameters":LOCAL_PARAMS, "runtime":"NumPy CPU (no PyTorch runtime)",
            "message":"Orbis-first model loaded; Qwen fallback is optional. See /docs."}

@ app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "runtime": "numpy-cpu",
        "parameters": LOCAL_PARAMS,
        "local_model": LOCAL_LABEL,
        "routing": "orbis-first",
        "qwen_fallback_enabled": bool(QWEN_FALLBACK_ENABLED and QWEN_FALLBACK_URL),
        "qwen_fallback_model": QWEN_FALLBACK_MODEL if QWEN_FALLBACK_ENABLED and QWEN_FALLBACK_URL else None,
        "legacy_backend_env_present_but_ignored": bool(os.getenv("ORBIS_BACKEND_URL") or os.getenv("ORBIS_PROVIDER")),
    }

@ app.get("/demo", response_class=HTMLResponse)
def demo():
    return """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Studio Orbis AI</title>
    <style>body{font:16px system-ui;max-width:760px;margin:40px auto;padding:0 16px;background:#10131a;color:#eef2ff}main{background:#1b2230;padding:22px;border-radius:16px}textarea{width:100%;min-height:90px;box-sizing:border-box;padding:12px;border-radius:10px;background:#111827;color:white}button{margin-top:10px;padding:11px 18px;border:0;border-radius:9px;background:#7c9cff;font-weight:700}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;padding:14px;border-radius:10px;min-height:60px}</style></head>
    <body><main><h1>Studio Orbis AI</h1><p>Step 7 Render server test</p><textarea id="m" placeholder="Ask Studio Orbis..."></textarea><br><button id="b">Send</button><h3>Response</h3><pre id="o">Ready.</pre></main>
    <script>b.onclick=async()=>{let m=document.getElementById('m').value.trim();if(!m)return;b.disabled=true;o.textContent='Generating...';try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});let d=await r.json();o.textContent=r.ok?d.response:(d.detail||JSON.stringify(d));}catch(e){o.textContent=String(e)}finally{b.disabled=false}}</script></body></html>"""

@ app.post("/chat")
def chat(request: ChatRequest):
    question = request.message.strip()

    # Explicit test hook: skip Orbis only when the caller deliberately asks to
    # test the backup. Normal requests always go to Orbis first.
    if request.force_qwen_fallback:
        try:
            backup = maybe_qwen_fallback(question, "manual_user_requested_fallback", request)
            if backup is None:
                raise HTTPException(status_code=503, detail="Qwen fallback is not configured. Set ORBIS_QWEN_URL to a running llama.cpp server.")
            return backup
        except HTTPException:
            raise
        except Exception as exc:
            error_detail = str(exc)
            if hasattr(exc, "response") and exc.response is not None:
                error_detail = f"HTTP {exc.response.status_code}: {exc.response.text[:1000]}"
            raise HTTPException(status_code=502, detail=f"Qwen fallback failed: {error_detail}")

    # Orbis is always the primary model. Do not use ORBIS_BACKEND_URL or a
    # provider preset to bypass it; this route intentionally ignores that legacy mode.
    try:
        answer = tiny_generate(
            question,
            request.max_new_tokens or 48,
            request.temperature,
            request.top_k,
        )
    except Exception as exc:
        try:
            backup = maybe_qwen_fallback(question, f"orbis_generation_error:{type(exc).__name__}", request)
            if backup is not None:
                return backup
        except Exception as backup_exc:
            raise HTTPException(status_code=502, detail=f"Orbis failed ({type(exc).__name__}) and Qwen fallback failed ({type(backup_exc).__name__}).")
        raise HTTPException(status_code=500, detail=f"Orbis generation failed: {type(exc).__name__}; Qwen fallback is not configured or unavailable.")

    bad, reason = looks_unusable(answer)
    if bad:
        try:
            backup = maybe_qwen_fallback(question, f"orbis_output:{reason}", request)
            if backup is not None:
                return backup
        except Exception as backup_exc:
            return {
                "response": answer or "Orbis returned an empty answer.",
                "route": "orbis-warning",
                "model": LOCAL_LABEL,
                "note": f"Orbis output failed basic checks ({reason}); Qwen fallback failed ({type(backup_exc).__name__}).",
            }
        return {
            "response": answer or "Orbis returned an empty answer.",
            "route": "orbis-warning",
            "model": LOCAL_LABEL,
            "note": f"Orbis output failed basic checks ({reason}); configure ORBIS_QWEN_URL to enable Qwen fallback.",
        }

    result = {
        "response": answer,
        "route": "orbis-primary",
        "model": LOCAL_LABEL,
    }
    if request.debug:
        result["quality_check"] = "passed_basic_checks"
        result["qwen_fallback_configured"] = bool(QWEN_FALLBACK_ENABLED and QWEN_FALLBACK_URL)
    return result

@ app.post("/generate")
def generate_endpoint(request: ChatRequest):
    result = chat(request)
    return {"text":result["response"]}