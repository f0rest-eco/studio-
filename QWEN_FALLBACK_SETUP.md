# Studio Orbis: Orbis-first, Qwen-second fallback

## Request flow

1. Every normal `/chat` request goes to the local Studio Orbis model first (`model2/` when loaded, otherwise Step 6).
2. The backend checks the generated text for empty output, malformed characters, symbol noise, and obvious repetition.
3. If Orbis throws an error or fails those checks, the backend sends the **original user question** to the configured Qwen endpoint.
4. The endpoint is expected to be an OpenAI-compatible `llama.cpp` server. The Qwen answer is returned to the website.

The quality check catches obvious gibberish, not fluent nonsense or factual errors. A response that passes the checks is not guaranteed to be correct.

## Important: Render does not run the Qwen model

Render hosts this lightweight Orbis FastAPI backend. The Qwen GGUF weights are not downloaded to Render. A separate computer/server must download the selected GGUF file from Hugging Face, run a compatible `llama.cpp` build, and expose its OpenAI-compatible API endpoint. The Render service connects to that endpoint over HTTPS.

The selected Hugging Face source is: https://huggingface.co/Jamesl0l/Qwen3.8-Flash-Next-GGUF

**Repository compatibility is not yet guaranteed.** Confirm the repository's exact GGUF filename, shard set, model architecture support, and minimum llama.cpp version before deployment. Do not download every quantization. Choose one complete quantization and all of its required shards.

## Render environment variables

Set these in Render → your service → Environment:

| Variable | Example | Meaning |
|---|---|---|
| `ORBIS_QWEN_URL` | `https://YOUR-QWEN-HOST/v1` | Base URL of the running llama.cpp server; do not include `/chat/completions` |
| `ORBIS_QWEN_MODEL` | `qwen3.8-flash-next` | Model name configured on the llama.cpp server |
| `ORBIS_QWEN_API_KEY` | optional secret | Only if your Qwen server requires bearer authentication |
| `ORBIS_QWEN_TIMEOUT` | `180` | Request timeout in seconds |
| `ORBIS_QWEN_ENABLED` | `true` | Enables/disables the backup |

If the Qwen host has no public endpoint yet, leave `ORBIS_QWEN_URL` unset. Orbis will continue to work; Qwen fallback cannot run until an endpoint is available.

### Keep Orbis first

This version's `/chat` handler always runs Orbis first. It intentionally ignores the old direct-backend routing path. Remove legacy `ORBIS_PROVIDER`, `ORBIS_BACKEND_URL`, and `ORBIS_GEMMA_MODEL` variables from Render to avoid confusion. Do not put a Qwen model URL in `ORBIS_BACKEND_URL`; use `ORBIS_QWEN_URL`.

## Run llama.cpp on a separate host

Install a recent llama.cpp build that explicitly supports the model architecture in the chosen repository. Download only the selected model file(s) from Hugging Face. The generic server command is:

```bash
llama-server \
  --model /path/to/selected/model.gguf \
  --host 0.0.0.0 \
  --port 8001 \
  --ctx-size 4096 \
  --alias qwen3.8-flash-next
```

If the quantization is split across multiple GGUF shards, use the first shard as the model path and keep all companion shards in the same directory, following that repository's instructions. Some architectures or MoE/embedding layouts may require a newer or specialized build; this command is a template, not proof that the linked repository will load.

Expose the endpoint securely over HTTPS, add authentication, and set `ORBIS_QWEN_URL` and `ORBIS_QWEN_API_KEY` in Render. Do not expose an unauthenticated inference server to the public internet.

## Test

Normal request (Orbis first; Qwen only if Orbis fails the basic quality check):

```bash
curl -X POST https://YOUR-ORbis-SERVICE.onrender.com/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Explain why the sky appears blue.","debug":true}'
```

Manual Qwen connectivity test (bypasses Orbis only for this explicit test):

```bash
curl -X POST https://YOUR-ORbis-SERVICE.onrender.com/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Say hello in one sentence.","force_qwen_fallback":true}'
```

`GET /health` reports whether the Qwen endpoint is configured. It does not prove the remote model is reachable or loaded.
