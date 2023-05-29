#! python3.7

import whisper
import torch

import numpy as np
import gradio as gr

demo = gr.Blocks(live=True)

audio_model = whisper.load_model("base")
language="English"

# Cue the user that we're ready to go.
print("Model loaded.\n")


def whisperRealtime(audio,state=""):

    result = audio_model.transcribe(audio=audio, fp16=torch.cuda.is_available(),language=language)
    text = result['text'].strip()

    state += text + " "
    return state, state


gr.Interface(
    fn=whisperRealtime, 
    inputs=[
        gr.Audio(source="microphone", type="filepath", streaming=True), 
        "state" 
    ],
     outputs=[
        "textbox",
        "state"
    ],
    live=True).queue().launch(debug=True,share=True)

