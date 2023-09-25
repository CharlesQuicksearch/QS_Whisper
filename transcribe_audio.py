import whisper

model = whisper.load_model("medium")

def transcribe_audiofile(audio_file):

    result = model.transcribe(audio_file, language="sv", fp16=False, verbose=True)
    result_text = result["text"]

    #with open("output.txt", "w", encoding="utf-8") as f:
    #    f.write(result_text)
    #    f.close()

    return result_text
