# for pip install openai whisper: pip install git+https://github.com/openai/whisper.git

from fastapi import FastAPI, HTTPException, UploadFile
from request_and_response import Response
import whisper
import tempfile
import shutil
import os

app = FastAPI()

# might want to let the incoming request send a value for which model to use
model = whisper.load_model("small")
#VRAM: tiny:1GB, base: 1GB, small:2GB, medium:5GB, large: 10GB

@app.get("/")
async def home():
    return {"Transcribe audio file. Make sure to send request content-type a multipart/form-data, and body key 'file' with the audio file as value."}

@app.post("/transcribe_audio/", response_model=Response)
async def upload_audio(file: UploadFile):
    try:
        #create a temporary directory and path to where the file is going
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio_file = os.path.join(temp_dir, file.filename)

            #open the path and copy the content from incoming request
            with open(temp_audio_file, "wb") as audio_file:
                shutil.copyfileobj(file.file, audio_file)

                #run the model with the temporary file
                result = model.transcribe(temp_audio_file, language="sv", fp16=False, verbose=True)
                result_text = result["text"]

        return Response(output=result_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

