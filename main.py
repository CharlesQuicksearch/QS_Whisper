# for pip install openai whisper: pip install git+https://github.com/openai/whisper.git
# might need blobfile

from fastapi import FastAPI, HTTPException, UploadFile
from request_and_response import Response
import whisper

import tempfile
import shutil
import os

app = FastAPI()
model = whisper.load_model("small") #VRAM GB: tiny:1, base: 1, small:2, medium:5

@app.get("/")
async def home():
    return {"Home of transcribe text"}

@app.post("/transcribe_audio/", response_model=Response)
async def upload_audio(file: UploadFile):
    try:

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio_file = os.path.join(temp_dir, file.filename)

            with open(temp_audio_file, "wb") as audio_file:
                shutil.copyfileobj(file.file, audio_file)

                result = model.transcribe(temp_audio_file, language="sv", fp16=False, verbose=True)
                result_text = result["text"]

        return Response(output=result_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

