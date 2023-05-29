from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch
import random

prompts = ["Porche 911 left", "Porche 911 right"]
#prompts = ["A bustling cityscape of manhattan, new york, in 1900", "A modern cityscape of manhattan, new york, in 2020"]

seeds = [random.randint(0, 9e9) for _ in range(len(prompts))]

# pipeline = StableDiffusionWalkPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     torch_dtype=torch.float32,
# )

pipeline = StableDiffusionWalkPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32,
)


video_path = pipeline.walk(
    prompts=prompts,
    seeds=seeds,
    fps=50,
    num_interpolation_steps=150,
    height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='dreams',        # Where images/videos will be saved
    #name='animals_test7',        # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=7,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=60,     # Number of diffusion steps per image generated. 50 is good default
)