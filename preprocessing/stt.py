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
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.get_chroma import get_or_create_user_chromadb
from preprocessing.split_and_store import split_student_answer

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


async def process_text(
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


class PreprocessVoiceRequest(BaseModel):
    user_id: str
    subject: str
    unit: str
    text: str  # raw STT text


@router.post("/preprocess_student_answer")
async def preprocess_student_answer(req: PreprocessVoiceRequest) -> JSONResponse:
    try:
        vectorstore = get_or_create_user_chromadb(req.user_id)
        correct_text = await process_text(
            text=req.text,
            subject=req.subject,
            unit=req.unit,
            vectorstore=vectorstore,
        )
        return JSONResponse(
            content={
                "success": True,
                "content": correct_text,
            }
        )
    except Exception as e:
        logger.exception("Unexpected error during voice preprocessing.")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error"},
        )
