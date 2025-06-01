"""
챗봇 프롬프트: 질문 + 관련 answer_key + 관련 feedback
=> 프롬프팅을 통해 LLM이 데이터를 구분하고, 질문 유형(내용/위치/피드백)에 맞는 응답 생성
"""

import os
from operator import itemgetter
from dotenv import load_dotenv

# langchain 관련 묶음
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

# 기타 외부 모듈
from langchain_teddynote import logging

load_dotenv()
logging.langsmith("Beakji-chat")

username = "user123"
persist_directory = os.getenv("PERSIST_DIRECTORY")


vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings(),
    collection_name=username,
)

retriever = vectordb.as_retriever()
answer_key_retriever = vectordb.as_retriever(
    search_kwargs={"filter": {"type": "answer_key"}}
)
feedback_retriever = vectordb.as_retriever(
    search_kwargs={"filter": {"type": "feedback"}}
)

prompt = PromptTemplate.from_template(
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

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# LangChain Runnable 구성
chain = (
    {
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
        "answer_key": itemgetter("question") | answer_key_retriever,
        "feedback": itemgetter("question") | feedback_retriever,
    }
    | prompt  # 프롬프트 적용
    | llm  # LLM 호출
    # StrOutputParser 생략: ChatMessageHistory에서 메시지 그대로 사용하기 위함
)

# 세션 기록을 저장할 딕셔너리
# -> DB에서 세션 기록 받아와야 함
store = {}


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 대화를 기록하는 RAG 체인 생성
rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)


if __name__ == "__main__":
    rag_with_history.invoke(
        # 질문 입력
        {"question": "베게너가 주장한 판구조론은?"},
        # 세션 ID 기준으로 대화를 기록합니다.
        config={"configurable": {"session_id": "rag123"}},
    )

    rag_with_history.invoke(
        # 질문 입력
        {"question": "그 내용 교안의 어디에 있어?"},
        # 세션 ID 기준으로 대화를 기록합니다.
        config={"configurable": {"session_id": "rag123"}},
    )

    rag_with_history.invoke(
        # 질문 입력
        {"question": "판구조론에서 내가 틀렸던 내용이 뭐였지?"},
        # 세션 ID 기준으로 대화를 기록합니다.
        config={"configurable": {"session_id": "rag123"}},
    )
