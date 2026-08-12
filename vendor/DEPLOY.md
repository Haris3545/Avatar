# Deploying our patched fork of fofr/face-to-many

`vendor/cog-face-to-many` is a fork of https://github.com/fofr/cog-face-to-many
with one fix: the original `custom_lora_url` validation only accepted LoRA
weights hosted at `replicate.delivery/pbxt/...`, but Replicate now routes
different accounts to different CDN bucket segments (ours is `xezq`, not
`pbxt`) -- so the hosted model rejects our correctly-working trained LoRA
URL. The patch (see `predict.py`, `parse_custom_lora_url` and the validation
block in `predict()`) generalizes both to accept any `replicate.delivery`
URL ending in `/trained_model.tar`, regardless of bucket segment. No other
behavior changes.

This needs to be built and pushed as your own Replicate model, since we
can't modify fofr's hosted copy. **This step can't run from this sandbox**
(no Docker, and Replicate's build/registry endpoints are network-blocked
here) -- it needs to run on your machine.

## Requirements

- Docker Desktop installed and running (https://www.docker.com/products/docker-desktop/)
- ~15-20GB free disk space (the image includes CUDA + torch + several ML
  libraries)
- Expect the build to take a while (looks like large images) — this is a
  one-time cost, not something you'll redo often
- No local GPU needed for the build itself — Cog builds the container image
  on CPU; the model only needs a GPU when it actually *runs* on Replicate

## Steps

```bash
# 1. Install Cog (Replicate's build tool)
brew install cog
# if brew doesn't have it, see https://github.com/replicate/cog#install

# 2. Log in with the same Replicate API token you've been using
cog login

# 3. Go to the forked model directory
cd vendor/cog-face-to-many

# 4. Push it to your own Replicate account as a new model
#    (create the destination model first at replicate.com/create,
#    same as we did for the LoRA training destination -- name it
#    something like "face-to-many-patched", set it Private)
cog push r8.im/haris3545/face-to-many-patched
```

`cog push` builds the Docker image locally and uploads it to Replicate,
which then hosts it on GPU hardware for actual predictions -- so build
time is mostly Docker layer downloads/installs, not GPU work.

Once it finishes, it'll print a version hash. Tell Claude that hash (or the
full `owner/model:version` string) and the generation script will be
updated to point at your patched model instead of `fofr/face-to-many`.
