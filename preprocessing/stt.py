"""
학생의 복습 음성 전처리

과정:
1. STT 수행 (Clova Speech API)
2. 맞춤법 교정 (LLM)
3. 결과 저장 및 반환
"""

import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from pydantic import BaseModel
import requests
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.get_chroma import get_or_create_user_chromadb
from preprocessing.split import split_student_answer

router = APIRouter()


class ClovaSpeechClient:
    def __init__(self):
        self.invoke_url = os.getenv("CLOVA_INVOKE_URL")
        self.secret = os.getenv("CLOVA_API_KEY")

    def req_upload(self, file, completion="sync") -> str:
        request_body = {
            "language": "ko-KR",
            "completion": completion,
            "wordAlignment": True,
            "fullText": True,
            "diarization": {"enable": False},
        }
        headers = {
            "Accept": "application/json;UTF-8",
            "X-CLOVASPEECH-API-KEY": self.secret,
        }
        with open(file, "rb") as f:
            files = {
                "media": f,
                "params": (
                    None,
                    json.dumps(request_body, ensure_ascii=False).encode("UTF-8"),
                    "application/json",
                ),
            }
            response = requests.post(
                url=self.invoke_url + "/recognizer/upload", headers=headers, files=files
            )
            return response.text


async def correct_typo(text: str) -> str:
    template = """The following is a student's test answer.

Please:
- Fix typos only.
- Do not add, remove, or rephrase content.
- Output in Korean only.

<Student answer>:
{content}

<Corrected answer>:
"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag_chain = prompt | model | StrOutputParser()
    return await rag_chain.ainvoke({"content": text})


async def voice_to_text(
    audio_file_path: str,
    subject: str,
    unit: str,
    vectorstore: Chroma,
    clova_client: ClovaSpeechClient,
) -> str:
    # 1. STT
    try:
        stt_result_json = clova_client.req_upload(audio_file_path, completion="sync")
        stt_text = json.loads(stt_result_json)["text"]
    except Exception as e:
        raise RuntimeError(f"STT failed: {e}")

    if not stt_text.strip():
        raise ValueError("STT returned empty result.")

    # 2. 맞춤법 교정
    try:
        corrected_text = await correct_typo(stt_text)
    except Exception as e:
        raise RuntimeError(f"Correcting typo failed: {e}")

    if not corrected_text.strip():
        raise ValueError("Correcting typo returned empty result.")

    # 3. 벡터 분할 및 Chroma DB 저장
    try:
        split_student_answer(vectorstore, subject, unit, corrected_text)
    except Exception as e:
        raise RuntimeError(f"Store in Chroma DB failed: {e}")

    return corrected_text


class PreprocessVoiceRequest(BaseModel):
    user_id: str
    subject: str
    unit: str
    audio_file_path: str


def get_chroma_db(req: PreprocessVoiceRequest = Depends()) -> Chroma:
    return get_or_create_user_chromadb(user_id=req.user_id)


def get_clova(req: PreprocessVoiceRequest = Depends()) -> ClovaSpeechClient:
    return ClovaSpeechClient()


@router.post("/preprocess_voice")
async def preprocess_voice(
    req: PreprocessVoiceRequest,
    vectorstore: Chroma = Depends(get_chroma_db),
    clova_client: ClovaSpeechClient = Depends(get_clova),
) -> JSONResponse:
    # 1. 음성 파일 존재 확인
    if not os.path.exists(req.audio_file_path):
        raise HTTPException(
            status_code=400, detail=f"Voice file doesn't exist: {req.audio_file_path}"
        )

    # 2. 음성 복습 전처리 (STT, 오타 정정, 벡터 분할 및 DB 저장)
    # return: 오타 정정 학생 답안
    try:
        correct_text = await voice_to_text(
            audio_file_path=req.audio_file_path,
            subject=req.subject,
            unit=req.unit,
            vectorstore=vectorstore,
            clova_client=clova_client,
        )
        if not correct_text.strip():
            raise HTTPException(status_code=422, detail="STT returned empty result.")
        return JSONResponse(content={"student_answer": correct_text})

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice preprocessing failed: {e}")
