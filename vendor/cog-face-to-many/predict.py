import os
import shutil
import random
import json
from PIL import Image, ExifTags
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

with open("face-to-many-api.json", "r") as file:
    workflow_json = file.read()


LORA_WEIGHTS_MAPPING = {
    "3D": "artificialguybr/3DRedmond-3DRenderStyle-3DRenderAF.safetensors",
    "Emoji": "fofr/emoji.safetensors",
    "Video game": "artificialguybr/PS1Redmond-PS1Game-Playstation1Graphics.safetensors",
    "Pixels": "artificialguybr/PixelArtRedmond-Lite64.safetensors",
    "Clay": "artificialguybr/ClayAnimationRedm.safetensors",
    "Toy": "artificialguybr/ToyRedmond-FnkRedmAF.safetensors",
}

LORA_TYPES = list(LORA_WEIGHTS_MAPPING.keys())


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.comfyUI.load_workflow(workflow_json, check_inputs=False)
        self.download_loras()

    def parse_custom_lora_url(self, url: str):
        # Original code assumed every account's LoRA weights are served from the
        # "pbxt" CDN bucket. Replicate now routes accounts to different bucket
        # segments (e.g. "xezq"), so parse generically: take whatever segment
        # sits directly before "/trained_model.tar", regardless of bucket name.
        path = url.split("://", 1)[-1]  # strip scheme
        parts = path.rstrip("/").split("/")
        # parts looks like [host, bucket, id, "trained_model.tar"] or
        # [host, id, "trained_model.tar"] for the pbxt.replicate.delivery form
        if parts[-1] == "trained_model.tar":
            parts = parts[:-1]
        return parts[-1]

    def add_to_lora_map(self, lora_url: str):
        uuid = self.parse_custom_lora_url(lora_url)
        self.comfyUI.weights_downloader.download_lora_from_replicate_url(uuid, lora_url)

    def download_loras(self):
        for weight in LORA_WEIGHTS_MAPPING.values():
            self.comfyUI.weights_downloader.download_weights(weight)

    def cleanup(self):
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_file(self, input_file: Path, prefix: str = "input"):
        file_extension = os.path.splitext(input_file)[1].lower()

        if file_extension in [".png", ".webp", ".gif"]:
            filename = f"{prefix}{file_extension}"
            shutil.copy(input_file, os.path.join(INPUT_DIR, filename))
        else:
            # If the file is jpg, jpeg or unknown, convert to png
            # We convert jpgs so that we remove any EXIF data related to HDR images – ComfyUI cannot load HDR images at the moment: https://github.com/fofr/cog-face-to-many/issues/1
            # Alternatively, if there is no file extension, like a base64 image
            # we'll try saving it as a PNG
            filename = f"{prefix}.png"
            image = Image.open(input_file)

            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(image._getexif().items())

                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
            except (KeyError, AttributeError):
                # EXIF data does not have orientation
                # Do not rotate
                pass

            image.save(os.path.join(INPUT_DIR, filename))

        return filename

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def update_workflow(self, workflow, **kwargs):
        style = kwargs["style"]
        prompt = kwargs["prompt"]
        negative_prompt = kwargs["negative_prompt"]
        custom_style = kwargs["lora_url"]

        if custom_style:
            uuid = self.parse_custom_lora_url(custom_style)
            lora_name = f"{uuid}/{uuid}.safetensors"
        else:
            lora_name = LORA_WEIGHTS_MAPPING[style]
            prompt = self.style_to_prompt(style, prompt)
            negative_prompt = self.style_to_negative_prompt(style, negative_prompt)

        load_image = workflow["22"]["inputs"]
        load_image["image"] = kwargs["filename"]

        # Node 100 feeds only the canny-edge ControlNet preprocessor (node
        # 49); node 22/67 (the real photo) still feeds ApplyInstantID (node
        # 41) for facial identity. Defaults to the same photo when no
        # separate control image is supplied, preserving old single-image
        # behaviour.
        control_load_image = workflow["100"]["inputs"]
        control_load_image["image"] = kwargs["control_filename"]

        loader = workflow["2"]["inputs"]
        loader["positive"] = prompt
        loader["negative"] = negative_prompt

        controlnet = workflow["28"]["inputs"]
        controlnet["strength"] = kwargs["control_image_strength"]

        lora_loader = workflow["3"]["inputs"]
        lora_loader["lora_name_1"] = lora_name
        lora_loader["lora_wt_1"] = kwargs["lora_scale"]

        instant_id = workflow["41"]["inputs"]
        instant_id["weight"] = kwargs["instant_id_strength"]

        sampler = workflow["4"]["inputs"]
        sampler["denoise"] = kwargs["denoising_strength"]
        sampler["seed"] = kwargs["seed"]
        sampler["cfg"] = kwargs["prompt_strength"]

        # When a mask is supplied, restrict generation to only the masked
        # region (node 104, VAEEncodeForInpaint) instead of the plain
        # VAEEncode (node 51) the sampler uses by default -- a ControlNet,
        # at any strength, only ever nudges the output toward the given
        # structure; it can't guarantee any of it survives. A hard mask is
        # the only way to actually lock everything outside it in place
        # (e.g. a scaffold's measured face/hair shape) while still letting
        # the model freely generate within the masked area. Node 104 takes
        # its base pixels from node 101 (the control_image branch), not
        # node 67 (the real photo) -- the protected, unmasked region must
        # show the scaffold's actual artwork, not a re-rendering of the
        # photo, so control_image should be the scaffold itself here.
        if kwargs.get("mask_filename"):
            mask_load_image = workflow["102"]["inputs"]
            mask_load_image["image"] = kwargs["mask_filename"]
            sampler["latent_image"] = ["104", 0]

    def style_to_prompt(self, style, prompt):
        style_prompts = {
            "3D": f"3D Render Style, 3DRenderAF, {prompt}",
            "Emoji": f"memoji, emoji, {prompt}, 3d render, sharp",
            "Video game": f"Playstation 1 Graphics, PS1 Game, {prompt}, Video game screenshot",
            "Pixels": f"Pixel Art, PixArFK, {prompt}",
            "Clay": f"Clay Animation, Clay, {prompt}",
            "Toy": f"FnkRedmAF, {prompt}, toy, miniature",
        }
        return style_prompts[style]

    def style_to_negative_prompt(self, style, negative_prompt=""):
        if negative_prompt:
            negative_prompt = f"{negative_prompt}, "

        start_base_negative = "nsfw, nude, oversaturated, "
        end_base_negative = "ugly, broken, watermark"
        specifics = {
            "3D": "photo, photography, ",
            "Emoji": "photo, photography, blurry, soft, ",
            "Video game": "text, photo, ",
            "Pixels": "photo, photography, ",
            "Clay": "",
            "Toy": "",
        }

        return f"{specifics[style]}{start_base_negative}{negative_prompt}{end_base_negative}"

    def predict(
        self,
        image: Path = Input(
            description="An image of a person to be converted",
            default=None,
        ),
        control_image: Path = Input(
            description="Optional separate image used only for canny-edge "
            "structure conditioning (e.g. a stylized line-art composite), while "
            "`image` still supplies facial identity via InstantID. If omitted, "
            "`image` is used for both. Ignored when `mask` is supplied.",
            default=None,
        ),
        mask: Path = Input(
            description="Optional inpaint mask (white = editable, black = kept "
            "exactly as in `image`). Restricts generation to the masked region "
            "only, guaranteeing everything outside it stays pixel-identical to "
            "`image` -- unlike control_image, which only ever nudges the output "
            "toward a structure without guaranteeing it survives.",
            default=None,
        ),
        style: str = Input(
            default="3D",
            choices=LORA_TYPES,
            description="Style to convert to",
        ),
        prompt: str = Input(default="a person"),
        negative_prompt: str = Input(
            default="",
            description="Things you do not want in the image",
        ),
        denoising_strength: float = Input(
            default=0.65,
            ge=0,
            le=1,
            description="How much of the original image to keep. 1 is the complete destruction of the original image, 0 is the original image",
        ),
        prompt_strength: float = Input(
            default=4.5,
            ge=0,
            le=20,
            description="Strength of the prompt. This is the CFG scale, higher numbers lead to stronger prompt, lower numbers will keep more of a likeness to the original.",
        ),
        control_image_strength: float = Input(
            default=0.8,
            ge=0,
            le=1,
            description="Strength of the canny-edge ControlNet conditioning on `control_image` "
            "(or `image` if `control_image` is not supplied). The bigger this is, the more "
            "strongly the traced edges affect the output.",
        ),
        instant_id_strength: float = Input(
            default=1, description="How strong the InstantID will be.", ge=0, le=1
        ),
        seed: int = Input(
            default=None, description="Fix the random seed for reproducibility"
        ),
        custom_lora_url: str = Input(
            default=None,
            description="URL to a Replicate custom LoRA. Must be in the format https://replicate.delivery/pbxt/[id]/trained_model.tar or https://pbxt.replicate.delivery/[id]/trained_model.tar",
        ),
        lora_scale: float = Input(
            default=1, description="How strong the LoRA will be", ge=0, le=1
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if image is None:
            raise ValueError("No image provided")

        filename = self.handle_input_file(image, prefix="face")
        control_filename = (
            self.handle_input_file(control_image, prefix="control")
            if control_image is not None
            else filename
        )
        mask_filename = (
            self.handle_input_file(mask, prefix="mask") if mask is not None else None
        )
        if custom_lora_url is not None:
            # Accept any replicate.delivery-hosted trained_model.tar, not just the
            # legacy "pbxt" bucket -- see parse_custom_lora_url for why.
            if (
                "replicate.delivery/" not in custom_lora_url
                or not custom_lora_url.endswith("/trained_model.tar")
            ):
                raise ValueError(
                    "Custom LoRA URL format is not supported. Must be a replicate.delivery URL "
                    "ending in /trained_model.tar"
                )
            self.add_to_lora_map(custom_lora_url)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        workflow = json.loads(workflow_json)
        self.update_workflow(
            workflow,
            filename=filename,
            control_filename=control_filename,
            style=style,
            denoising_strength=denoising_strength,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_strength=prompt_strength,
            instant_id_strength=instant_id_strength,
            lora_url=custom_lora_url,
            lora_scale=lora_scale,
            control_image_strength=control_image_strength,
            mask_filename=mask_filename,
        )

        wf = self.comfyUI.load_workflow(workflow, check_weights=False)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = []
        output_directories = [OUTPUT_DIR]

        for directory in output_directories:
            print(f"Contents of {directory}:")
            files.extend(self.log_and_collect_files(directory))

        return files
