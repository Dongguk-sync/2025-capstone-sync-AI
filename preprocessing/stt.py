"""
학생의 복습 음성 전처리

1. STT 수행 (Clova Speech API)
2. 맞춤법 교정 (LLM)
3. 결과 저장 및 반환
"""

import os
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from pydantic import BaseModel
import requests
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from preprocessing.split import split_student_answer

load_dotenv()
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
    vectorstore=Depends(get_chroma_db),
    clova_client=Depends(get_clova),
) -> JSONResponse:
    # 1. STT 요청
    try:
        stt_result_json = clova_client.req_upload(
            req.audio_file_path, completion="sync"
        )
        stt_text = json.loads(stt_result_json)["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 실패: {e}")

    if not stt_text.strip():
        raise HTTPException(status_code=422, detail="STT 결과가 비어 있습니다.")

    # 2. 맞춤법 교정
    try:
        corrected_text = await correct_typo(stt_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"맞춤법 교정 실패: {e}")

    if not corrected_text.strip():
        raise HTTPException(status_code=422, detail="교정된 텍스트가 비어 있습니다.")

    # 3. 청크 분할 및 저장
    try:
        split_student_answer(vectorstore, req.subject, req.unit, corrected_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma 저장 실패: {e}")

    # 4. 결과 반환
    return JSONResponse(content={"corrected_text": corrected_text})
