"""
교안 PDF 전처리

과정 : ocr -> markdown formatting -> chroma DB에 저장
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.get_chroma import get_or_create_user_chromadb
from preprocessing.split_and_store import split_answer_key

from config import (
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_STREAMING,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def markdown_formatting(text: str) -> str:
    template = """The following is a textbook content.

Please:
- Convert the content to markdown format in Korean.
- Extract **only the conceptual explanations** (not exercises, questions, or examples).
- Use '#' for the main title, '##' for section titles, and '###' for subsections.
- This is an answer key. Do not add, remove, or modify the content.

<content>:
{content}

<markdown>:
"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        streaming=OPENAI_STREAMING,
    )
    rag_chain = prompt | model | StrOutputParser()
    result = await rag_chain.ainvoke({"content": text})
    return result


async def prep_answer_key(
    text: str,
    subject: str,
    unit: str,
    vectorstore,
    url: Optional[str] = None,
) -> str:
    markdown_text = await markdown_formatting(text)
    if not markdown_text.strip():
        raise ValueError("Formatting returned empty result.")
    split_answer_key(
        vectorstore=vectorstore,
        subject=subject,
        unit=unit,
        text=markdown_text,
        url=url if url else None,
    )
    return markdown_text


class FormatAnswerKeyRequest(BaseModel):
    user_id: str
    subject: str
    unit: str
    text: str  # raw OCR text
    url: Optional[str] = None


@router.post("/answer_key")
async def format_answer_key(req: FormatAnswerKeyRequest) -> JSONResponse:
    try:
        vectorstore = get_or_create_user_chromadb(user_id=req.user_id)

        markdown_text = await prep_answer_key(
            text=req.text,
            subject=req.subject,
            unit=req.unit,
            vectorstore=vectorstore,
            url=req.url,
        )

        return JSONResponse(
            content={
                "success": True,
                "answer_key": markdown_text,
            }
        )
    except ValueError as ve:
        logger.warning(f"Text preprocessing validation failed: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("Text preprocessing failed.")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error"},
        )
