"""
교안 PDF를 입력 받았을 시:
OCR -> markdown_formatting -> ChromaDB 저장 (answer_key)

복습 음성을 입력 받았을 시:
STT -> correct_typo -> ChromaDB 저장 (student_answer)

채점시:
answer_key, student_answer -> compare_evaluate -> ChromaDB 저장 (Feedback)
"""

from fastapi import FastAPI
from preprocessing import ocr, stt
from mychain import chat, evaluate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.include_router(ocr.router, prefix="/ocr")
app.include_router(stt.router, prefix="/stt")
app.include_router(chat.router, prefix="/chat")
app.include_router(evaluate.router, prefix="/evaluate")
