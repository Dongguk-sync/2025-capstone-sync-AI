"""
교안 열람에서 채팅
프롬프트: 질문 + answer_key + student_answer + feedback + chat history
"""

import logging
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from enum import Enum

# langchain 관련 묶음
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# 기타 외부 모듈
from langchain_teddynote import logging as langchain_logging

from config import (
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_STREAMING,
)


logger = logging.getLogger(__name__)

router = APIRouter()
langchain_logging.langsmith("Beakji-chat")


async def get_chat_response(
    question: str,
    file_content: str,
    studys_stt_content: str,
    studys_feed_content: str,
    history_obj: ChatMessageHistory,
):
    try:
        prompt = PromptTemplate.from_template(
            """You are an assistant for learning review.
Use the following information to answer the question as helpfully as possible.
Respond in the following JSON format:
{{
  "message_content": "<natural language answer>",
  "message_created_at": "<response DATETIME>"
}}

#Previous Chat History:
{chat_history}

#Answer Key:
{file_content}

#Student STT Answer:
{studys_stt_content}

#Student Feedback:
{studys_feed_content}

#Question:
{question}

#Answer:
"""
        )

        prompt_input = prompt.format(
            question=question,
            file_content=file_content,
            studys_stt_content=studys_stt_content,
            studys_feed_content=studys_feed_content,
            chat_history=str(history_obj.messages),
        )

        model = ChatOpenAI(
            model_name=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            streaming=OPENAI_STREAMING,
        )

        result = await model.ainvoke(prompt_input)

        if isinstance(result, BaseMessage):
            result = result.content

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
    subject_name: str
    file_name: str
    file_content: str
    studys_stt_content: str
    studys_feed_content: str
    chat_bot_history_id: str
    chat_bot_history: list[ChatMessage]


@router.post("/lecture_chat")
async def chat(req: ChatRequest) -> JSONResponse:
    try:
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
            req.file_content,
            req.studys_stt_content,
            req.studys_feed_content,
            history_obj,
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
                    "subject_name": req.subject_name,
                    "file_name": req.file_name,
                },
            }
        )

    except HTTPException as he:
        logger.warning(f"[HTTPException] {he.detail}")
        return JSONResponse(
            status_code=he.status_code,
            content={"success": False, "error": he.detail},
        )
    except Exception as e:
        logger.exception("Unexpected error during chat request.")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error"},
        )
