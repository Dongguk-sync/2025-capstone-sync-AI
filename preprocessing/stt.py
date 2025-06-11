"""
학생의 복습 음성 전처리

과정:
1. STT 수행 (Clova Speech API)
2. 맞춤법 교정 (LLM)
3. 결과 저장 및 반환
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from pydantic import BaseModel
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.get_chroma import get_or_create_user_chromadb
from utils.split_and_store import split_student_answer

from config import (
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_STREAMING,
)

logger = logging.getLogger(__name__)

router = APIRouter()


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
    model = ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        streaming=OPENAI_STREAMING,
    )
    rag_chain = prompt | model | StrOutputParser()
    return await rag_chain.ainvoke({"content": text})


async def prep_student_answer(
    text: str,
    subject: str,
    unit: str,
    vectorstore: Chroma,
) -> str:
    try:
        corrected_text = await correct_typo(text)
    except Exception as e:
        raise RuntimeError(f"Correcting typo failed: {e}")

    if not corrected_text.strip():
        raise ValueError("Correcting typo returned empty result.")

    try:
        split_student_answer(vectorstore, subject, unit, corrected_text)
    except Exception as e:
        raise RuntimeError(f"Store in Chroma DB failed: {e}")

    return corrected_text


class FormatAnswerRequest(BaseModel):
    user_id: str
    subject: str
    unit: str
    text: str  # raw STT text


@router.post("/student_answer")
async def format_student_answer(req: FormatAnswerRequest) -> JSONResponse:
    try:
        vectorstore = get_or_create_user_chromadb(req.user_id)
        correct_text = await prep_student_answer(
            text=req.text,
            subject=req.subject,
            unit=req.unit,
            vectorstore=vectorstore,
        )
        return JSONResponse(
            content={
                "success": True,
                "student_answer": correct_text,
            }
        )
    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        return JSONResponse(
            status_code=422, content={"success": False, "error": str(ve)}
        )
    except RuntimeError as re:
        logger.error(f"Processing error: {re}")
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(re)}
        )
    except Exception as e:
        logger.exception("Unexpected error during voice preprocessing.")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error"},
        )
