# Deploying our patched fork of fofr/face-to-many

`vendor/cog-face-to-many` is a fork of https://github.com/fofr/cog-face-to-many
with two fixes:

1. The original `custom_lora_url` validation only accepted LoRA weights
   hosted at `replicate.delivery/pbxt/...`, but Replicate now routes
   different accounts to different CDN bucket segments (ours is `xezq`, not
   `pbxt`) -- so the hosted model rejects our correctly-working trained
   LoRA URL. The patch (see `predict.py`, `parse_custom_lora_url` and the
   validation block in `predict()`) generalizes both to accept any
   `replicate.delivery` URL ending in `/trained_model.tar`, regardless of
   bucket segment.

2. A new optional `control_image` input. Previously the single `image`
   input fed both InstantID's facial identity embedding *and* the depth
   ControlNet's structure conditioning, since both derived from the same
   workflow node. That meant we couldn't swap in a separately-generated
   structure map (e.g. `scripts/composite_line_art.py`'s output) without
   also breaking face detection -- InstantID needs a real photo to find a
   face in, and a pure line-art composite has none. The patch adds a
   second `LoadImage` + resize branch (nodes 100/101 in
   `face-to-many-api.json`) that feeds only the depth preprocessor (node
   49); `image` still feeds InstantID (via node 67) for identity. When
   `control_image` is omitted, it defaults to the same photo as `image`,
   so old single-image behavior is unchanged.

This needs to be built and pushed as your own Replicate model, since we
can't modify fofr's hosted copy. Building it needs Docker, which isn't
available either in this sandbox (network-blocked from Replicate's
registry) or apparently on your machine (can't install Docker Desktop) --
so **this runs in GitHub Actions instead**, which gives us a Docker-capable
machine with unrestricted internet, without touching your laptop at all.

## One-time setup

1. **Create the destination model on Replicate** (same as we did for LoRA
   training): go to [replicate.com/create](https://replicate.com/create),
   name it e.g. `face-to-many-patched`, set it **Private**.

2. **Get a Cog CLI auth token** (different from the `r8_...` API token
   you've been using) at
   [replicate.com/auth/token](https://replicate.com/auth/token).

3. **Add it as a GitHub Actions secret**: on the repo on GitHub, go to
   Settings → Secrets and variables → Actions → New repository secret.
   Name it `REPLICATE_CLI_AUTH_TOKEN`, paste the token from step 2.

## Running the deploy

Go to the repo on GitHub → **Actions** tab → **"Deploy patched
face-to-many to Replicate"** in the left sidebar → **Run workflow** button.
Confirm the `model_name` input matches what you created in step 1 (e.g.
`haris3545/face-to-many-patched`), then run it.

It'll build the image (several GB of CUDA/torch/ML dependencies -- expect
it to take a while) and push it to Replicate. Watch the run's logs in the
Actions tab; once it finishes it'll print a version hash in the `cog push`
step output. That's what gets plugged into the generation script next.

`cog push` builds the Docker image locally and uploads it to Replicate,
which then hosts it on GPU hardware for actual predictions -- so build
time is mostly Docker layer downloads/installs, not GPU work.

Once it finishes, it'll print a version hash. Tell Claude that hash (or the
full `owner/model:version` string) and the generation script will be
updated to point at your patched model instead of `fofr/face-to-many`.
