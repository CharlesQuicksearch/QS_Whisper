# for pip install openai whisper: pip install git+https://github.com/openai/whisper.git
# might need blobfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from transcribe_audio import transcribe_audiofile
from request_and_response import Response

app = FastAPI()

@app.get("/")
async def home():
    return {"Home of transcribe text"}

@app.post("/transcribe_audio/", response_model=Response)
async def upload_audio(file: UploadFile):
    try:
        result = await transcribe_audiofile(file)
        return Response(output=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

