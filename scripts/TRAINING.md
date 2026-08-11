# Training the VCCP avatar style LoRA

Style-only LoRA, trained on the 33 avatars in `data/avatars` (no headshot pairs
needed — see the pipeline discussion for why). Capped at 4 training runs to
keep spend under ~£4-5.

## 1. One-time setup

```
pip install -r scripts/requirements.txt
export REPLICATE_API_TOKEN=r8_...   # from https://replicate.com/account/api-tokens
```

You'll need a Replicate account with prepaid credits (no free tier).

## 2. Prepare the dataset

```
python3 scripts/prepare_lora_dataset.py
```

Composites all avatars onto a white background, center-crops to square,
resizes to 1024x1024, captions each with the shared style description, and
zips the result to `data/lora_dataset.zip`. Re-run this if you add/remove
avatars before a training run.

## 3. Train

```
python3 scripts/train_lora.py --destination yourusername/vccp-avatar-lora
```

- Trains an SDXL LoRA on `stability-ai/sdxl`, ~10-15 min, ~$0.60-1/run.
- Every run is logged to `data/training_runs.json`. The script refuses to
  start a 5th run — delete/edit that file if you deliberately want to spend
  more.
- Tune between runs with `--steps` (default 1000) and `--lr` (default 1e-4)
  if the style isn't converging - e.g.:
  ```
  python3 scripts/train_lora.py --destination yourusername/vccp-avatar-lora --steps 1500 --lr 5e-5 --notes "more steps, lower lr"
  ```

## 4. Test

Once a run finishes, Replicate hosts the trained model at your `--destination`.
Generate a test avatar with a ControlNet (e.g. `lllyasviel/sd-controlnet-canny`
or an SDXL ControlNet variant) feeding edges extracted from a real headshot,
combined with the trained LoRA, prompted with the `TOK style` trigger phrase
used during training.
