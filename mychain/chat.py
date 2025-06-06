"""
챗봇 프롬프트: 질문 + 관련 answer_key + 관련 feedback
=> 프롬프팅을 통해 LLM이 데이터를 구분하고, 질문 유형(내용/위치/피드백)에 맞는 응답 생성
"""

import json
from operator import itemgetter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# langchain 관련 묶음
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory.chat_message_histories import ChatMessageHistory

# 기타 외부 모듈
from langchain_teddynote import logging
from utils.get_chroma import get_or_create_user_chromadb

from config import (
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_STREAMING,
)

router = APIRouter()
logging.langsmith("Beakji-chat")


def get_chat_prompt():
    return PromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
Return only a valid JSON object.
Return your answer in JSON format with the following structure:
{{
  "answer": "<natural language answer>",
  "url": "<answer key URL if available, otherwise null>"
}}
If the answer key is not relevant, set the url to null.
If you don't know the answer, set both fields to null.

#Previous Chat History:
{chat_history}

#Question: 
{question} 

#Context: 
- Answer Key:
{answer_key}

- Feedback:
{feedback}

#Answer (in JSON):"""
    )


def build_rag_chain(answer_key_retriever, feedback_retriever):
    return (
        {
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
            "answer_key": itemgetter("question") | answer_key_retriever,
            "feedback": itemgetter("question") | feedback_retriever,
        }
        | get_chat_prompt()
        | ChatOpenAI(
            model_name=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            streaming=OPENAI_STREAMING,
        )
    )


async def get_chat_response(
    question: str,
    history_obj: ChatMessageHistory,
    history_id: str,
    answer_key_retriever,
    feedback_retriever,
):
    try:
        rag_chain = build_rag_chain(answer_key_retriever, feedback_retriever)

        rag_with_history = RunnableWithMessageHistory(
            rag_chain,
            lambda _: history_obj,  # ChatMessageHistory 객체 반환
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        result = await rag_with_history.ainvoke(
            {"question": question},
            config={"configurable": {"session_id": history_id}},  # ✅ 반드시 session_id
        )

        # LLM이 BaseMessage 객체로 반환되었을 경우 content 추출
        if isinstance(result, BaseMessage):
            result = result.content

        # 응답이 문자열이면 JSON 파싱
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500, detail=f"LLM response is not valid JSON: {e}"
                )

        # 딕셔너리 아니면 에러
        if not isinstance(result, dict) or "answer" not in result:
            raise HTTPException(
                status_code=500, detail="Invalid response format from LLM."
            )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat response failed: {e}")


# 요청 모델
class ChatMessage(BaseModel):
    message_type: str  # "user" or "AI"
    message_content: str


class ChatRequest(BaseModel):
    question: str
    user_id: str
    history_id: str
    chat_history: list[ChatMessage]


def get_vectorstores(user_id: str):
    try:
        vectordb = get_or_create_user_chromadb(user_id=user_id)
        answer_key_retriever = vectordb.as_retriever(
            search_kwargs={"filter": {"type": "answer_key"}}
        )
        feedback_retriever = vectordb.as_retriever(
            search_kwargs={"filter": {"type": "feedback"}}
        )
        return answer_key_retriever, feedback_retriever
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VectorStore loading failed: {e}")


@router.post("/chat")
async def chat(req: ChatRequest) -> JSONResponse:
    try:
        answer_key_retriever, feedback_retriever = get_vectorstores(req.user_id)

        # ChatMessageHistory 객체 생성
        history_obj = ChatMessageHistory()
        for msg in req.chat_history:
            if msg.message_type == "user":
                history_obj.add_user_message(msg.message_content)
            elif msg.message_type == "AI":
                history_obj.add_ai_message(msg.message_content)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid message_type: {msg.message_type}",
                )

        # ChatMessageHistory 객체 전달
        result = await get_chat_response(
            req.question,
            history_obj,
            req.history_id,
            answer_key_retriever,
            feedback_retriever,
        )

        if not isinstance(result, dict) or "answer" not in result:
            raise HTTPException(
                status_code=500, detail="Invalid response format from LLM."
            )

        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
