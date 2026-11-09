# Training your own Orbis model (no API, fully homemade)

`train.py` trains a Studio Orbis v2-style transformer from scratch in PyTorch and exports a
`model2/` folder the NumPy server serves directly (app.py picks it up automatically).

## Sizes

| preset | params | NumPy speed (laptop CPU) | Colab T4 pretrain time |
|---|---|---|---|
| small  | ~7M    | very fast | ~6-10 h per 100M tokens |
| medium | ~17M   | ~20-60 ms/token | ~10-16 h per 100M tokens |
| large  | ~33M   | ~50-120 ms/token | needs several days / Pro |

## Step 1 — pretrain (Colab)

1. Upload this project to Colab (or GitHub) and set Runtime -> T4 GPU.
2. `pip install torch datasets tokenizers`
3. Run:  `python train.py --size medium --tokens 150000000 --out runs/orbis_m`
   - Data: streams FineWeb-Edu from Hugging Face (~150M tokens ≈ one long Colab session).
   - Checkpoints + `model2/`-format export are written every 10k steps; you can resume with `--init runs/orbis_m/ckpt.pt`.
   - Watch `val_loss`: expect it to land around **3.2-3.6** for medium. That is a coherent English model.

## Step 2 — teach it to chat (SFT)

Build a jsonl file with one JSON object per line: {"user": "...", "assistant": "..."}.
Good free sources: a 20-50k subset of OpenHermes-2.5, UltraChat, or No Robots
(reformat their fields into user/assistant). Then:

    python train.py --size medium --sft chat.jsonl --tokens 20000000 \
        --init runs/orbis_m/ckpt.pt --out runs/orbis_m_sft --lr 1e-4

Loss is computed only on the assistant's tokens, and the prompt uses the same
`User: ...\nAssistant:` template that app.py now sends at inference time.

## Step 3 — deploy

Copy the exported `runs/orbis_m_sft/` folder over the `model2/` folder in this project
(config.json + orbis_weights.npz + tokenizer.json), redeploy, and `/chat` answers with your
own model. `/health` shows `local_model` and the parameter count.

## Honest expectations

- A well-trained 17-33M model writes grammatical, on-topic paragraphs and can follow simple
  chat instructions. It will not do reliable multi-step reasoning, math, or up-to-date facts.
- The quality ladder is roughly: your 1.1M untrained model (current) < 7M trained < 17M trained
  << 124M (GPT-2 class) << 1B+ << frontier models. Each rung costs ~10x more compute.
- To go further at home: a Kaggle P100 session (30h/week free) roughly doubles throughput,
  and quantization/fp16 NumPy serving (ORBIS weights stay fp32 in npz) can be added later.
