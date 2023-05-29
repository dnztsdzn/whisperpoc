import numpy as np
import gradio as gr

demo = gr.Blocks(live=True)

import whisper

base_model = whisper.load_model("medium")

import gradio as gr 
import time
import openai
import pytube as pt

import os, sys, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from transformers import logging

from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch
import random

openai.api_key = "api-key"

# Choose model to use by uncommenting
#modelName = "tiny"
#modelName = "base.en"
#modelName = "small.en"
#modelName = "medium.en"
modelName = "medium"
#modelName = "large-v2"

# def translateAndAI(audio):
    
#     time.sleep(3)
#     # load audio and pad/trim it to fit 30 seconds
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)

#     # make log-Mel spectrogram and move to the same device as the model
#     mel = whisper.log_mel_spectrogram(audio).to(base_model.device)

#     # decode the audio
#     options = whisper.DecodingOptions(without_timestamps=True,fp16 = False, task='translate')
#     result = whisper.decode(base_model, mel, options)

#     #result = base_model.transcribe(audio, task = 'translate')
#     print(result.text)

#     m = GPT4All()
#     m.open()
#     out = m.prompt(result.text)
#     print(out)
#     return "user: "+ result.text+ "\n" + "ai: " + out


def stableDiffusionVideo(prompts,fps,num_interpolation_steps,height,width,guidance,inference):
    #prompts = ["A bustling cityscape of Istanbul, Turkey, in 1900", "A modern cityscape of Istanbul, Turkey, in 2020"]

    prompts = prompts.split(";")

    seeds = [random.randint(0, 9e9) for _ in range(len(prompts))]

    pipeline = StableDiffusionWalkPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
    )

    video_path = pipeline.walk(
    prompts=prompts,
    seeds=seeds,
    fps=int(fps),
    num_interpolation_steps=int(num_interpolation_steps),
    height=int(height),  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=int(width),   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    #output_dir='dreams',        # Where images/videos will be saved
    #name='animals_test7',        # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=int(guidance),         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=int(inference),     # Numbe r of diffusion steps per image generated. 50 is good default
    )



    return video_path
    #visualize_video_colab(video_path)
    


def translateAndAI(audio,textTest):
    
    whisperResult = ""
    if audio is not None:

        model = whisper.load_model(modelName)
        result = model.transcribe(audio=audio, word_timestamps=False, verbose=True, task='translate')
        whisperResult = result["text"]

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user", "content": textTest + "; "+ whisperResult}
    ]
    )

    return "user: "+ textTest+"; "+ whisperResult + "\n" + "\n" + "ai: " + completion["choices"][0]["message"]["content"]

def transcribe(audio):
    
    model = whisper.load_model(modelName)
    result = model.transcribe(audio=audio, word_timestamps=False, verbose=True)
    print(result["text"])

    return result["text"]

def translate(audio):
    
    model = whisper.load_model(modelName)
    result = model.transcribe(audio=audio, word_timestamps=False, verbose=True, task='translate')
    print(result["text"])

    return result["text"]

def downloadYoutube(url):
    yt = pt.YouTube(url)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(
        output_path="./mp3",
        filename=yt.title+".mp3"
    )



#! python
# copyright (c) 2022 C.Y. Wong, myByways.com simplified Stable Diffusion v0.1



def stableDiffusionTest(audio,textInput2):

    whisperResult = ""

    if audio is not None:
        model = whisper.load_model(modelName)
        result = model.transcribe(audio=audio, word_timestamps=False, verbose=True)
        whisperResult = result["text"]


    print(textInput2+whisperResult+"TESTTTTTTTTTTTT")

    PROMPTS.clear()

    PROMPTS.append(textInput2+whisperResult)

    print('*** Loading Stable Diffusion - myByways.com simple-sd version 0.1')
    tic1 = time.time()
    logging.set_verbosity_error()
    os.makedirs(FOLDER, exist_ok=True)

    seed_pre()
    config = OmegaConf.load(CONFIG)
    model = load_model(config)
    device, precision_scope = set_device(model)
    sampler = setup_sampler(model)
    start_code = seed_post(device)

    toc1 = time.time()
    print(f'*** Model setup time: {(toc1 - tic1):.2f}s')

    counter = 0
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():

                for iteration in range(ITERATIONS):
                    for prompt, negative in get_prompts():
                        print(f'*** Iteration {iteration + 1}: {prompt}')
                        tic2 = time.time()
                        images = generate_samples(model, sampler, prompt, negative, start_code)
                        for image in images:
                            name = save_image(image)
                            save_history(name, prompt, negative)
                            print(f'*** Saved image: {name}')
                        toc2 = time.time()

                        print(f'*** Synthesis time: {(toc2 - tic2):.2f}s')
                        counter += len(images)

    print(f'*** Total time: {(toc2 - tic1):.2f}s')
    print(f'*** Saved {counter} image(s) to {FOLDER} folder.')

    image = 255. * rearrange(images[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(image.astype(np.uint8))

    
    return img


PROMPTS = [         # --prompt, one or more in an array
    
]
NEGATIVES = [       # negative prompt, one or more, default None (or an empty array)
    'rover'
]

HEIGHT = 512        # --H, default 512, beyond causes M1 to crawl
WIDTH = 512         # --W, default 512, beyond causes M1 to crawl
FACTOR = 8          # --f downsampling factor, default 8

FIXED = 0           # --fixed_code, 1 for repeatable results, default 0
SEED = 42           # --seed, default 42
NOISE = 0.0         # --ddim_eta, 0 deterministic, no noise - 1.0 random noise, ignored for PLMS (must be 0)
PLMS = 0            # --plms, default 1 on M1 for txt2img but ignored for img2img (must be DDIM)
ITERATIONS = 1      # --n_iter, default 1
SCALE = 7.5         # --scale, 5 further -> 15 closer to prompt, default 7.5
STEPS = 50          # --ddim_steps, practically little improvement >50 but takes longer, default 50

IMAGE = None        # --init-img, img2img initial latent seed, default None
STRENGTH = 0.75     # --strength 0 more -> 1 less like image, default 0.75

FOLDER = 'outputs'  # --outdir for images and history file below
HISTORY = 'history.txt'
CONFIG = 'v1-inference.yaml'
CHECKPOINT = 'sd-v1-4.ckpt'

def seed_pre():
    if not FIXED:
        seed_everything(SEED)

def seed_post(device):
    if FIXED:
        seed_everything(SEED)
        return torch.randn([1, 4, HEIGHT // FACTOR, WIDTH // FACTOR], device='cpu').to(torch.device(device.type))
    return None

def load_model(config, ckpt=CHECKPOINT):
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    return model

def set_device(model):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        precision = nullcontext
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        precision = torch.autocast
    else:
        device = torch.device('cpu')
        precision = torch.autocast
    model.to(device.type)
    model.eval()
    return device, precision

def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.0

def setup_sampler(model):
    global NOISE
    if IMAGE:
        image = load_image(IMAGE).to(model.device.type)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(image))
        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=STEPS, ddim_eta=NOISE, verbose=False)
        t_enc = int(STRENGTH * STEPS)
        sampler.t_enc = t_enc
        sampler.z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(model.device.type))
    elif PLMS:
        sampler = PLMSSampler(model)
        NOISE = 0
    else:
        sampler = DDIMSampler(model)
    return sampler

def get_prompts():
    global NEGATIVES
    if NEGATIVES is None:
        NEGATIVES = [''] * len(PROMPTS)
    else:
        NEGATIVES.extend([''] * (len(PROMPTS)-len(NEGATIVES)))
    return zip(PROMPTS, NEGATIVES)

def generate_samples(model, sampler, prompt, negative, start):
    uncond = model.get_learned_conditioning(negative) if SCALE != 1.0 else None
    cond = model.get_learned_conditioning(prompt)
    if IMAGE:
        samples = sampler.decode(sampler.z_enc, cond, sampler.t_enc, 
            unconditional_guidance_scale=SCALE, unconditional_conditioning=uncond)
    else:
        shape = [4, HEIGHT // FACTOR, WIDTH // FACTOR]
        samples, _ = sampler.sample(S=STEPS, conditioning=cond, batch_size=1,
            shape=shape, verbose=False, unconditional_guidance_scale=SCALE, 
            unconditional_conditioning=uncond, eta=NOISE, x_T=start)
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    return x_samples

def save_image(image):
    name = f'{time.strftime("%Y%m%d_%H%M%S")}.png'
    image = 255. * rearrange(image.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(image.astype(np.uint8))
    img.save(os.path.join(FOLDER, name))
    return name

def save_history(name, prompt, negative):
    with open(os.path.join(FOLDER, HISTORY), 'a') as history:
        history.write(f'{name} -> {"PLMS" if PLMS else "DDIM"}, Seed={SEED}{" fixed" if FIXED else ""}, Scale={SCALE}, Steps={STEPS}, Noise={NOISE}')
        if IMAGE:
            history.write(f', Image={IMAGE}, Strength={STRENGTH}')
        if len(negative):
            history.write(f'\n + {prompt}\n - {negative}\n')
        else:
            history.write(f'\n + {prompt}\n')



with demo:
    gr.Markdown("Speech recognition whisper model: medium")
    with gr.Tab("Microphone"):
        with gr.Column():
            with gr.Row():
                microphone_input = gr.inputs.Audio(source="microphone", type="filepath")
                microphoneChatgptTextInput = gr.inputs.Textbox(label="Prompt for chatgpt")
                microphoneStableDiffusionTextInput = gr.inputs.Textbox(label="Prompt for image generation")
            with gr.Row():
                microphoneTranscriptionOutput = gr.Textbox(label="Transcription")
                microphoneTranslateOutput = gr.Textbox(label="Translation")
                microphoneTranslateAndAiOutput = gr.Textbox(label="Chatgpt")
                image_output = gr.Image()
            with gr.Row():
                micTranscribeButton = gr.Button("Transcribe")
                micTranslateButton = gr.Button("Translate")
                micChatgptButton = gr.Button("Chatgpt")
                micStableDiffusionButton= gr.Button("Stable diffusion")
    with gr.Tab("File"):
        with gr.Column():
            with gr.Row():
                file_input = gr.inputs.Audio(source="upload", type="filepath")
                fileChatgptTextInput = gr.inputs.Textbox(label="Prompt for chatgpt")
                fileStableDiffusionTextInput = gr.inputs.Textbox(label="Prompt for image generation")
            with gr.Row():
                fileTranscriptionOutput = gr.Textbox(label="Transcription")
                fileTranslateOutput = gr.Textbox(label="Translation")
                fileTranslateAndAiOutput = gr.Textbox(label="Chatgpt")
                image_output_2 = gr.Image()
            with gr.Row():
                fileTranscribeButton = gr.Button("Transcribe")
                fileTranslateButton = gr.Button("Translate")
                fileChatgptButton = gr.Button("Chatgpt")
                fileStableDiffusionButton = gr.Button("Stable diffusion")
    with gr.Tab("Stable Diffusion Video"):
        with gr.Column():
            with gr.Row():
                videoFps = gr.inputs.Number(label="Fps",default="2")
                videoSteps = gr.inputs.Number(label="Number of steps",default="2")
                videoGuidance = gr.inputs.Textbox(label="Guidance", default="10")
                videoInference = gr.inputs.Number(label="Inference",default="50")
                videoHeight = gr.inputs.Number(label="Height",default="512")
                videoWidth = gr.inputs.Number(label="Width",default="512")
                videoPrompts = gr.inputs.Textbox(label="Prompts", default="cat; dog")
                videoOutput = gr.Video(format='mp4')
            with gr.Row():
                videoGenerateButton = gr.Button("Generate")
    # with gr.Tab("Youtube"):
    #     with gr.Column():
    #         with gr.Row():
    #             youtube_input = gr.inputs.Textbox(label="Youtube link")
    #             fileChatgptTextInput = gr.inputs.Textbox(label="Prompt for chatgpt")
    #             fileStableDiffusionTextInput = gr.inputs.Textbox(label="Prompt for image generation")
    #         with gr.Row():
    #             fileTranscriptionOutput = gr.Textbox(label="Transcription")
    #             fileTranslateOutput = gr.Textbox(label="Translation")
    #             fileTranslateAndAiOutput = gr.Textbox(label="Chatgpt")
    #             image_output_2 = gr.Image()
    #         with gr.Row():
    #             urlTranscribeButton = gr.Button("Transcribe")
    #             urlTranslateButton = gr.Button("Translate")
    #             urlChatgptButton = gr.Button("Chatgpt")
    #             urlStableDiffusionButton = gr.Button("Stable diffusion")


    micTranscribeButton.click(transcribe, inputs=microphone_input, outputs=microphoneTranscriptionOutput)
    micTranslateButton.click(translate, inputs=microphone_input, outputs=microphoneTranslateOutput)
    micChatgptButton.click(translateAndAI, inputs=[microphone_input,microphoneChatgptTextInput], outputs=microphoneTranslateAndAiOutput)
    #text_button3.click(translateAndAI, inputs=microphone_input, outputs=text_outputTranslateAndAi)
    micStableDiffusionButton.click(stableDiffusionTest, inputs=[microphone_input,microphoneStableDiffusionTextInput], outputs=image_output)

    fileTranscribeButton.click(transcribe, inputs=file_input, outputs=fileTranscriptionOutput)
    fileTranslateButton.click(translate, inputs=file_input, outputs=fileTranslateOutput)
    fileChatgptButton.click(translateAndAI, inputs=[file_input,fileChatgptTextInput], outputs=fileTranslateAndAiOutput)
    #text_button3_2.click(translateAndAI, inputs=text_input2, outputs=text_outputTranslateAndAi_2)
    fileStableDiffusionButton.click(stableDiffusionTest, inputs=[file_input,fileStableDiffusionTextInput], outputs=image_output_2)

    videoGenerateButton.click(stableDiffusionVideo, inputs=[videoPrompts,videoFps,videoSteps,videoHeight,videoWidth,videoGuidance,videoInference], outputs=videoOutput)

    # urlTranscribeButton.click(transcribe, inputs=file_input, outputs=fileTranscriptionOutput)
    # urlTranslateButton.click(translate, inputs=file_input, outputs=fileTranslateOutput)
    # urlChatgptButton.click(translateAndAI, inputs=[file_input,fileChatgptTextInput], outputs=fileTranslateAndAiOutput)
    # #text_button3_2.click(translateAndAI, inputs=text_input2, outputs=text_outputTranslateAndAi_2)
    # urlStableDiffusionButton.click(stableDiffusionTest, inputs=[file_input,fileStableDiffusionTextInput], outputs=image_output_2)

demo.queue().launch(debug=True,share=True)