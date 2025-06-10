import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from utils.get_chroma import get_or_create_user_chromadb
from langchain_teddynote import logging as langchain_logging

from config import (
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_STREAMING,
)

logger = logging.getLogger(__name__)
langchain_logging.langsmith(project_name="Beakji-evaluate")

router = APIRouter()


# 요청 모델
class EvaluationRequest(BaseModel):
    user_id: str
    subject: str
    unit: str
    answer_key_text: str
    student_answer_text: str


# 프롬프트 템플릿
def get_evaluation_prompt():
    return ChatPromptTemplate.from_template(
        """Evaluate <student answer> based on the <answer key>.

        - Feedback is based on the answer key.
        - Don't evaluate information that's not in the answer key.
        - Focus on conceptual accuracy rather than wording
        - Recognize that foreign proper nouns may be spelled differently.
        - Please write in markdown format and Korean.
        - Organize your feedback under these 3 headings: 
          - # 복습 피드백: {unit}
          - ## 누락 내용 (Missing)
          - ## 틀린 내용 (Incorrect)

        <Answer key>:
        {answer_key}

        <Student answer>:
        {student_answer}
        """
    )


# 평가 로직
async def get_evaluation_result(
    vectorstore,
    answer_key_text: str,
    student_answer_text: str,
    subject: str,
    unit: str,
) -> str:
    model = ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        streaming=OPENAI_STREAMING,
    )

    rag_chain = (
        RunnablePassthrough() | get_evaluation_prompt() | model | StrOutputParser()
    )

    # 비동기 평가 수행
    result = await rag_chain.ainvoke(
        {
            "unit": unit,
            "answer_key": answer_key_text,
            "student_answer": student_answer_text,
        }
    )

    # 결과 저장
    vectorstore.add_texts(
        texts=[result],
        metadatas=[{"subject": subject, "unit": unit, "type": "feedback"}],
    )

    return result


# 평가 API 비동기 엔드포인트
@router.post("/evaluate")
async def evaluate(
    req: EvaluationRequest,
) -> JSONResponse:
    try:
        vectorstore = get_or_create_user_chromadb(req.user_id)
        feedback = await get_evaluation_result(
            vectorstore=vectorstore,
            answer_key_text=req.answer_key_text,
            student_answer_text=req.student_answer_text,
            subject=req.subject,
            unit=req.unit,
        )
        return JSONResponse(
            content={
                "success": True,
                "content": {
                    "subject": req.subject,
                    "unit": req.unit,
                    "feedback": feedback,
                },
            }
        )
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
            },
        )
