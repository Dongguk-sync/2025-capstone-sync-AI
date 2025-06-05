"""
교안 PDF 전처리

과정 : ocr -> markdown formatting -> chroma DB에 저장
"""

import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from pdf2image import convert_from_path
from pydantic import BaseModel
import pytesseract

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.get_chroma import get_or_create_user_chromadb
from preprocessing.split import split_answer_key

router = APIRouter()


class Ocr:
    def __init__(self):
        self.poppler_path = os.getenv("POPPLER_PATH")

    def ocr_from_pdf(self, pdf_path, dpi=300, lang="kor+eng") -> str:
        try:
            images = convert_from_path(
                pdf_path, dpi=dpi, poppler_path=self.poppler_path
            )
        except Exception as e:
            raise RuntimeError(f"PDF 변환 실패: {e}")

        all_text = ""
        for image in images:
            text = pytesseract.image_to_string(image, lang=lang)
            all_text += f"\n{text}"
        return all_text


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
    model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)
    rag_chain = prompt | model | StrOutputParser()
    result = await rag_chain.ainvoke({"content": text})
    return result


async def pdf_to_text(
    pdf_path: str,
    subject: str,
    unit: str,
    vectorstore: Chroma,
    ocr: Ocr,
    url: Optional[str] = None,
) -> str:
    # 1. OCR
    try:
        raw_text = ocr.ocr_from_pdf(pdf_path)
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}")

    # 2. 마크다운 변환
    try:
        markdown_text = await markdown_formatting(raw_text)
    except Exception as e:
        raise RuntimeError(f"Markdown formatting failed: {e}")

    # 3. 벡터 분할 및 Chroma DB 저장
    try:
        split_answer_key(
            text=markdown_text,
            subject=subject,
            unit=unit,
            url=url,
            vectorstore=vectorstore,
        )
    except Exception as e:
        raise RuntimeError(f"Store in Chroma DB failed: {e}")

    return markdown_text


class PreprocessPdfRequest(BaseModel):
    user_id: str
    subject: str
    unit: str
    pdf_path: str
    url: Optional[str] = None


def get_chroma_db(req: PreprocessPdfRequest = Depends()) -> Chroma:
    return get_or_create_user_chromadb(user_id=req.user_id)


def get_ocr(req: PreprocessPdfRequest = Depends()) -> Ocr:
    return Ocr()


@router.post("/preprocess_pdf")
async def preprocess_pdf(
    req: PreprocessPdfRequest,
    vectorstore: Chroma = Depends(get_chroma_db),
    ocr: Ocr = Depends(get_ocr),
) -> JSONResponse:
    # 1. PDF 파일 존재 확인
    if not os.path.exists(req.pdf_path):
        raise HTTPException(
            status_code=404,
            detail=f"PDF file doesn't exist: {req.pdf_path}",
        )

    # 2. PDF 교안 전처리 (OCR, 마크다운 형식, 벡터 분할 및 DB 저장)
    # return: 마크다운 형식 정답
    try:
        markdown_text = await pdf_to_text(
            pdf_path=req.pdf_path,
            subject=req.subject,
            unit=req.unit,
            url=req.url,
            vectorstore=vectorstore,
            ocr=ocr,
        )
        if not markdown_text.strip():
            raise HTTPException(status_code=422, detail="OCR returned empty result.")
        return JSONResponse(content={"answer_key": markdown_text})

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF preprocessing failed: {e}")
