# NEZ Young

**NEZ Young** is a tiny decoder-only language model trained **entirely from scratch** (random initialization, no pretrained weights) on a 5,705-example tool-use corpus.

This repository demonstrates training a language model using only GitHub Actions CPU runners and pure PyTorch.

## Dataset

- File: [`data/nez_young_tooluse_corpus.json`](data/nez_young_tooluse_corpus.json)
- 5,705 examples covering tool use, reasoning, code, finance, etc.
- Splits: train / validation / test
- Each example contains: `user_prompt`, `thought_process`, `tool_calls`, `tool_outputs`, `final_response`

## Model

- Architecture: GPT-style decoder-only Transformer
- From-scratch random init (normal 0.02)
- Size (default config):
  - 4 layers
  - 4 attention heads
  - 128-dim embeddings
  - Character-level vocabulary (built from the corpus)
  - Context length 256
- ~0.5–1M parameters (fits easily on CPU)

## Training

Training is performed via GitHub Actions:

1. Go to the **Actions** tab
2. Select **Train NEZ Young from Scratch**
3. Click **Run workflow**

Or just push changes to `main` that touch the training script / data.

The workflow:

- Installs PyTorch
- Runs `train_from_scratch.py` (2 000 iterations, ~few minutes on CPU)
- Uploads checkpoints as artifacts
- Optionally commits the best checkpoint if it is small enough

### Local training

```bash
pip install -r requirements.txt
python train_from_scratch.py
```

## Inference (after training)

```python
import torch, json
from train_from_scratch import NEZYoung, Config, decode, encode  # or load the class

ckpt = torch.load("checkpoints/nez_young_best.pt", map_location="cpu")
cfg = Config()
# rebuild model with saved config...
model = NEZYoung(cfg)
model.load_state_dict(ckpt["model"])
model.eval()

prompt = "### User\nWhat's the current price of Bitcoin?\n\n### Thought\n"
idx = torch.tensor([encode(prompt)], dtype=torch.long)
out = model.generate(idx, max_new_tokens=150)
print(decode(out[0].tolist()))
```

## Notes

- This is an educational / demonstration model. Character-level modeling + tiny capacity means it will not produce high-quality long-form text.
- The point is to show a complete, reproducible **from-scratch** training pipeline that runs on free GitHub Actions resources.
- No external pretrained weights are ever loaded.

## License

MIT (model + training code). Dataset is provided as-is for research / educational use.
