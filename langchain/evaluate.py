from fastapi import APIRouter, Depends
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from get_chroma import get_or_create_user_chromadb
from langchain_teddynote import logging

# 환경 설정
import os
from dotenv import load_dotenv

load_dotenv()
logging.langsmith(project_name=os.getenv("LANGSMITH_PROJECT"))

router = APIRouter()


# 요청 모델
class EvaluationRequest(BaseModel):
    user_id: str
    subject: str
    unit: str
    answer_key_text: str
    student_answer_text: str


# Chroma 의존성 주입
def get_chroma_db(req: EvaluationRequest = Depends()):
    return get_or_create_user_chromadb(user_id=req.user_id)


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


# 평가 로직 (async)
async def get_evaluation_result_async(
    vectorstore,
    answer_key_text: str,
    student_answer_text: str,
    subject: str,
    unit: str,
) -> str:
    model = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0)

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
    vectorstore=Depends(get_chroma_db),
):
    feedback = await get_evaluation_result_async(
        vectorstore=vectorstore,
        answer_key_text=req.answer_key_text,
        student_answer_text=req.student_answer_text,
        subject=req.subject,
        unit=req.unit,
    )
    return {"feedback": feedback}
