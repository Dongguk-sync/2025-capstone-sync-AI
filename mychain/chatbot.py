"""
챗봇 프롬프트: 질문 + 관련 answer_key + 관련 feedback
=> 프롬프팅을 통해 LLM이 데이터를 구분하고, 질문 유형(내용/위치/피드백)에 맞는 응답 생성
"""

import logging
import json
from operator import itemgetter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from enum import Enum

# langchain 관련 묶음
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 기타 외부 모듈
from langchain_teddynote import logging as langchain_logging
from utils.get_chroma import get_or_create_user_chromadb

from config import (
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_STREAMING,
)


logger = logging.getLogger(__name__)

router = APIRouter()
langchain_logging.langsmith("Beakji-chat")


def get_chat_prompt():
    return PromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
Return only a valid JSON object.
Return your answer in JSON format with the following structure:
{{
  "message_content": "<natural language answer>",
  "message_created_at": "<response DATETIME>
  "file_url": "<answer key URL if available, otherwise null>"
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
            config={"configurable": {"session_id": history_id}},
        )

        retrieved_docs = await answer_key_retriever.aget_relevant_documents(question)

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

        if not isinstance(result, dict) or "message_content" not in result:
            raise HTTPException(
                status_code=500, detail="Invalid response format from LLM."
            )

        subject_name = None
        file_name = None
        if retrieved_docs:
            meta = retrieved_docs[0].metadata
            subject_name = meta.get("subject")
            file_name = meta.get("unit")

        result["subject_name"] = subject_name
        result["file_name"] = file_name

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat response failed: {e}")


class MessageType(str, Enum):
    HUMAN = "HUMAN"
    AI = "AI"


class ChatMessage(BaseModel):
    message_type: MessageType  # "HUMAN" or "AI"
    message_content: str


class ChatRequest(BaseModel):
    question: str
    user_id: str
    chat_bot_history_id: str
    chat_bot_history: list[ChatMessage]
    subject_name: str
    file_name: str


def get_retrievers(user_id: str):
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


@router.post("/chatbot")
async def chat(req: ChatRequest) -> JSONResponse:
    try:
        answer_key_retriever, feedback_retriever = get_retrievers(req.user_id)

        # ChatMessageHistory 객체 생성
        history_obj = ChatMessageHistory()
        for msg in req.chat_bot_history:
            if msg.message_type == MessageType.HUMAN:
                history_obj.add_user_message(msg.message_content)
            elif msg.message_type == MessageType.AI:
                history_obj.add_ai_message(msg.message_content)
            else:
                raise HTTPException(
                    status_code=400, detail=f"Invalid message_type: {msg.message_type}"
                )

        # ChatMessageHistory 객체 전달
        result = await get_chat_response(
            req.question,
            history_obj,
            req.chat_bot_history_id,
            answer_key_retriever,
            feedback_retriever,
        )

        if not isinstance(result, dict) or "message_content" not in result:
            raise HTTPException(
                status_code=500, detail="Invalid response format from LLM."
            )

        return JSONResponse(
            content={
                "success": True,
                "content": {
                    "message_type": MessageType.AI,
                    "message_content": result.get("message_content"),
                    "message_created_at": result.get("message_created_at"),
                    "subject_name": result.get("subject_name", req.subject_name),
                    "file_name": result.get("file_name", req.file_name),
                    "file_url": result.get("file_url"),
                },
            }
        )

    except HTTPException as he:
        logger.warning(f"[HTTPException] {he.detail}")
        return JSONResponse(
            status_code=he.status_code,
            content={
                "success": False,
                "error": he.detail,
            },
        )
    except Exception as e:
        logger.exception("Unexpected error during chat request.")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
            },
        )
