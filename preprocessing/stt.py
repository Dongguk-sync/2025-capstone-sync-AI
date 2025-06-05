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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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
        files = {
            "media": open(file, "rb"),
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
    template = """- Fix typos
    - The content is an answer to a test and should not be added to or deleted.
    - Write in Korean

    <content>:
    {content}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
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
        raise RuntimeError(f"STT 실패: {e}")

    if not stt_text.strip():
        raise ValueError("STT 결과가 비어 있습니다.")

    # 2. 맞춤법 교정
    try:
        corrected_text = await correct_typo(stt_text)
    except Exception as e:
        raise RuntimeError(f"맞춤법 교정 실패: {e}")

    if not corrected_text.strip():
        raise ValueError("맞춤법 교정 결과가 비어 있습니다.")

    # 3. 벡터 분할 및 Chroma DB 저장
    try:
        split_student_answer(vectorstore, subject, unit, corrected_text)
    except Exception as e:
        raise RuntimeError(f"Chroma 저장 실패: {e}")

    # 4. 결과 반환
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
            status_code=400, detail=f"파일이 존재하지 않습니다: {req.audio_file_path}"
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
            raise HTTPException(status_code=422, detail="STT 결과가 비어 있습니다.")
        return JSONResponse(content={"student_anser": correct_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
