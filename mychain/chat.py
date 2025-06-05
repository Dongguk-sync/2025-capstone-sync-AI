"""
챗봇 프롬프트: 질문 + 관련 answer_key + 관련 feedback
=> 프롬프팅을 통해 LLM이 데이터를 구분하고, 질문 유형(내용/위치/피드백)에 맞는 응답 생성
"""

from operator import itemgetter

# langchain 관련 묶음
import langchain_chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

# 기타 외부 모듈
from langchain_teddynote import logging
from utils.get_chroma import get_or_create_user_chromadb

from fastapi import FastAPI, APIRouter, Depends, Request, Body
from pydantic import BaseModel

logging.langsmith("Beakji-chat")


def get_chat_prompt():
    return PromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. 
    If the user asks for the location of the answer key for that content, return the “url” of a vector with the extracted metadatas as “type”:"answer_key”.
    If you don't know the answer, just say that you don't know.

    #Previous Chat History:
    {chat_history}

    #Question: 
    {question} 

    #Context: 
    - Answer Key:
    {answer_key}

    - Feedback:
    {feedback}

    #Answer:"""
    )


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
# -> 임시로 세팅해둔 것. 서버 요청하여 DB에서 받아오기
_session_store = {}


def get_session_history(session_id):
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def build_rag_chain(answer_key_retriever, feedback_retriever):
    return (
        {
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
            "answer_key": itemgetter("question") | answer_key_retriever,
            "feedback": itemgetter("question") | feedback_retriever,
        }
        | get_chat_prompt()
        | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    )


def get_chat_response(
    question: str, session_id: str, answer_key_retriever, feedback_retriever
):
    rag_chain = build_rag_chain(answer_key_retriever, feedback_retriever)

    rag_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return rag_with_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )


class ChatRequest(BaseModel):
    question: str
    session_id: str
    user_id: str


def get_vectorstores(user_id: str):
    vectordb = get_or_create_user_chromadb(user_id=user_id)
    answer_key_retriever = vectordb.as_retriever(
        search_kwargs={"filter": {"type": "answer_key"}}
    )
    feedback_retriever = vectordb.as_retriever(
        search_kwargs={"filter": {"type": "feedback"}}
    )
    return answer_key_retriever, feedback_retriever


app = FastAPI()
router = APIRouter()


@router.post("/chat")
def chat(
    chat_request: ChatRequest = Body(...),
    retrievers=Depends(
        lambda chat_request=Body(...): get_vectorstores(chat_request.user_id)
    ),
):
    answer_key_retriever, feedback_retriever = retrievers
    result = get_chat_response(
        chat_request.question,
        chat_request.session_id,
        answer_key_retriever,
        feedback_retriever,
    )
    return {"answer": result}


app.include_router(router)
