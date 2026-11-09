# Studio Orbis AI — Combined Step 10

This is a single deployable FastAPI project containing both homemade Orbis checkpoints plus an optional Gemma model loader and the existing optional OpenAI-compatible backend.

- Base install: `pip install -r requirements.txt`
- Optional Gemma install on a capable inference machine: `pip install -r requirements-gemma.txt`
- Configure `ORBIS_GEMMA_MODEL` with a local model directory or Hugging Face repository ID.
- See `GEMMA_SETUP.md` before uploading large weights.

The two homemade checkpoints are preserved in their own folders because their tokenizers and network shapes differ. Combining them into one file of weights would corrupt both models. This package combines their inference service and adds a model-selection path instead.

Render deployment remains Python/no-Docker: `uvicorn app:app --host 0.0.0.0 --port $PORT`. Do not add multi-gigabyte weights to the Git repository. Host them in a Hugging Face model repository or object storage and load them from the server.
