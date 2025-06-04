"""
교안 PDF 전처리
과정 : ocr -> markdown formatting -> chroma DB에 저장
"""

import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from pdf2image import convert_from_path
from pydantic import BaseModel
import pytesseract

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from preprocessing.split import _store_chunks

load_dotenv()

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


async def markdown_formatting(text) -> str:
    template = """- Convert the given content to markdown format in korean
    - The given text is from a textbook. Extract only the concepts, not the exercises in the textbook.
    - Use # for the entire article, ## the medium topic, and ### for the small topic.
    - The article is an answer key to a test and should not be added to or deleted.

    <content>:
    {content}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
    result = await rag_chain.ainvoke({"content": text})
    return result


async def pdf_to_text(
    pdf_path: str,
    vectorstore: Chroma,
    subject: str,
    unit: str,
    doc_type: str = "answer_key",
    url: Optional[str] = None,
) -> str:
    # 1. OCR
    ocr = Ocr()
    raw_text = ocr.ocr_from_pdf(pdf_path)

    # 2. 마크다운 변환
    markdown_text = await markdown_formatting(raw_text)

    # 3. 벡터스토어 저장
    _store_chunks(
        vectorstore=vectorstore,
        chunks=[markdown_text],  # 현재는 한 개의 큰 청크로 저장
        subject=subject,
        unit=unit,
        doc_type=doc_type,
        url=url,
    )

    return markdown_text


class PreprocessPdfRequest(BaseModel):
    user_id: str
    subject: str
    unit: str
    pdf_path: str


def get_chroma_db(req: PreprocessPdfRequest = Depends()) -> Chroma:
    return get_or_create_user_chromadb(user_id=req.user_id)


@router.post("/preprocess")
async def preprocess_pdf(
    req: PreprocessPdfRequest,
    vectorstore=Depends(get_chroma_db),
):
    # 파일 존재 확인
    if not os.path.exists(req.pdf_path):
        raise HTTPException(
            status_code=404,
            detail=f"PDF 파일이 존재하지 않습니다: {req.pdf_path}",
        )

    try:
        markdown_text = await pdf_to_text(
            pdf_path=req.pdf_path,
            vectorstore=vectorstore,
            subject=req.subject,
            unit=req.unit,
            doc_type="answer_key",
        )

        if not markdown_text.strip():
            raise HTTPException(status_code=422, detail="OCR 결과가 비어 있습니다.")

        return JSONResponse(content={"markdown": markdown_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 전처리 실패: {e}")
