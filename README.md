# Studio Orbis — Orbis-first with optional Qwen fallback

This FastAPI project is intended for GitHub → Render deployment. Render runs the lightweight NumPy-based Studio Orbis model. A separate, suitably provisioned machine may run the Qwen GGUF backup through llama.cpp.

## Routing behavior

- Normal requests always go to Studio Orbis first.
- The backend checks for empty, malformed, symbol-heavy, or obviously repetitive output.
- Only when Orbis throws an exception or fails those basic checks does the backend send the original question to Qwen.
- The caller can explicitly set `force_qwen_fallback: true` to test Qwen connectivity; this is a diagnostic override, not the normal route.
- If no Qwen endpoint is configured, Orbis remains available and the API reports that fallback is unavailable.

These checks cannot reliably detect fluent nonsense or factual errors.

## Deploy to Render

1. Push the contents of this `Orbis_AI` directory to your GitHub repository (project root should contain `app.py`, `requirements.txt`, `render.yaml`, and model folders).
2. Connect the repository to Render using the existing Blueprint or Web Service setup in `render.yaml`.
3. Keep the Python build command `pip install --no-cache-dir -r requirements.txt` and start command `uvicorn app:app --host 0.0.0.0 --port $PORT`.
4. Remove legacy `ORBIS_PROVIDER`, `ORBIS_BACKEND_URL`, and `ORBIS_GEMMA_MODEL` variables from Render. This version enforces Orbis-first routing.
5. Test `/health`, `/docs`, and `/chat`.
6. Only after a separate Qwen inference server is running, set `ORBIS_QWEN_URL` and optional `ORBIS_QWEN_API_KEY` in Render.

See [`QWEN_FALLBACK_SETUP.md`](QWEN_FALLBACK_SETUP.md) for the separate inference-server requirements and test commands.

## Model limitations

The included Orbis model is a small experimental model. The free Render instance has limited CPU/RAM and may sleep after inactivity. The Qwen GGUF repository is a model-file source, not a running endpoint. The selected Qwen quantization is large and requires a separate host with enough RAM/VRAM, storage, and a compatible llama.cpp build. It is not practical to load the full Qwen checkpoint on Render Free.
