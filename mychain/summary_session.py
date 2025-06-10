from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_STREAMING,
)

router = APIRouter()


class SummarySessionRequest(BaseModel):
    first_question: str


def get_summary_prompt():
    return ChatPromptTemplate.from_template(
        """
        Below is the first user question from a chatbot session.

        - Summarize the user's initial intent into a short title.
        - Use a brief noun phrase, not a full sentence.
        - Respond in Korean.
        - Do not include any assistant replies.
        - The title must be no longer than 15 Korean characters.

        <User Question>
        {first_question}

        <Summary>:
        """
    )


async def summarize_first_question(first_question: str) -> str:
    model = ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        streaming=OPENAI_STREAMING,
    )

    rag_chain = get_summary_prompt() | model | StrOutputParser()
    result = await rag_chain.ainvoke({"first_question": first_question})
    return result


@router.post("/summary_session")
async def summarize_session(req: SummarySessionRequest) -> JSONResponse:
    try:
        summary = await summarize_first_question(req.first_question)
        return JSONResponse(
            content={
                "success": True,
                "summary": summary,
            }
        )
    except ValueError as ve:
        return JSONResponse(
            status_code=422, content={"success": False, "error": str(ve)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error"},
        )
