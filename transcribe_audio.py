import whisper
import numpy as np

model = whisper.load_model("tiny")

async def transcribe_audiofile(audio):

    #audio_array = np.frombuffer(audio_data, dtype=np.int32)

    result = await model.transcribe(audio, language="sv", fp16=False, verbose=True)
    result_text = result["text"]

    return result_text
