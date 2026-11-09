# Gemma add-on for Studio Orbis AI

This package combines the two existing homemade Orbis checkpoints in one service and adds an optional Gemma inference path. The Orbis checkpoints remain separate: their architectures and tokenizers are different from Gemma, so their weight tensors cannot be safely averaged or concatenated into Gemma weights.

## Add Gemma weights without GitHub

**Recommended: Hugging Face Hub model repository.** Create a model repository at https://huggingface.co/new (public or private), upload the Gemma checkpoint files and the matching tokenizer files there, and keep the model files in that repository. Large model files are supported by the Hub's large-file storage. For a private repository, create a read token and add it as the `HF_TOKEN` secret in your hosting service.

Set these environment variables in Render (or your inference host):

- `ORBIS_GEMMA_MODEL` = `account-name/repository-name` (or a mounted local model directory)
- `HF_TOKEN` = a Hugging Face read token only if the repository is private/gated

The model repository must contain a Transformers-compatible Gemma checkpoint and its matching tokenizer. A web page URL, Kaggle preview URL, or GitHub page URL is not a model identifier. Do not mix a Gemma 3 checkpoint with Gemma 4 config/tokenizer files or with the Keras Gemma 3 files; the architecture and tokenizer format must match the weights exactly.

Other non-GitHub storage choices include Cloudflare R2, Amazon S3, Google Cloud Storage, and Backblaze B2. Those are object-storage buckets, not code repositories. They work best when the server downloads model shards to persistent storage at startup or mounts a persistent volume. For sharded checkpoints, keep every shard and the index file together. Do not put private download links or tokens in frontend JavaScript.

## Runtime

The default `requirements.txt` stays small and supports the existing NumPy Orbis models. To enable Gemma, install `requirements-gemma.txt` on a separate inference server with enough memory, then set `ORBIS_GEMMA_MODEL`. The app will use Gemma when no OpenAI-compatible backend is configured. If `ORBIS_BACKEND_URL` is set, the existing backend smart-routing path remains active.

A 1B model generally needs multiple GB of memory even when its weights file is around 2 GB. Render's free instance and the user's Chromebook (about 2.7 GiB RAM) are not appropriate places to load this full model. Keep the model files in Hugging Face or object storage, and run inference on a host with sufficient RAM/accelerator memory. Render can continue hosting the lightweight Orbis API/frontend and call the separate inference endpoint through a private server-side backend setting.

## Files supplied with the project

- `model/`: original homemade Step 6 checkpoint and tokenizer.
- `model2/`: homemade Orbis v2 checkpoint and its distinct tokenizer.
- `app.py`: one API that can use either homemade checkpoint, an optional Gemma model, or the configured OpenAI-compatible backend.

The extra `config.json` supplied alongside the Kaggle Gemma 3 link declares `Gemma4ForConditionalGeneration` / `model_type: gemma4`. It is not a Gemma 3 config and is intentionally not installed as the runtime config. Use the exact config and tokenizer that belong to the actual weights you upload.
