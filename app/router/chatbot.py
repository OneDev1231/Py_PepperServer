from fastapi import APIRouter, UploadFile, File
from app.utils.func import answer_question, transcribe_audio

import os
import shutil

router = APIRouter()

@router.get("/user-question")
# def answer_user_question(question: Question_Model):
def answer_user_question(msg: str):
    try:
        return answer_question(msg)
    except Exception as e:
        print(e)
        return e

@router.post("/transcribe")
def whipser(file: UploadFile = File(...)):
    shutil.rmtree('data')
    os.makedirs('data', exist_ok=True)
    with open(f'data/{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return transcribe_audio(file.filename)


