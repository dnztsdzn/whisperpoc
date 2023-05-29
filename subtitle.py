import whisper
import gradio as gr
from whisper.utils import WriteSRT
import tempfile
import ffmpeg
from pathlib import Path
from whisper.utils import get_writer

whisperResult=""

def asdtest(audio):
    model = whisper.load_model("medium")
    result = model.transcribe(audio=audio, verbose=True, task='translate')
    whisperResult = result["text"]

    p = Path(audio)
    writer = WriteSRT(p.parent)
    writer(result, audio,options=1)



gr.Interface(fn=asdtest, inputs=gr.Video(source="upload", type="filepath"), outputs=gr.Video()).launch()