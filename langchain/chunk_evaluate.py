# Step 0: Set env
import langchain
import os

from langchain_teddynote import logging

logging.langsmith("Beakji-evaluate")

from dotenv import load_dotenv

load_dotenv()
persist_directory = os.getenv("PERSIST_DIRECTORY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from split import format_docs
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def chunk_evaluate(vectorstore: Chroma, subject: str, unit: str, answer_key_chunk: str):

    # 벡터 db에서 정답과 관련된 답안 청크 가져오기
    results = vectorstore.similarity_search_with_score(
        answer_key_chunk,  # 정답 청크
        k=3,  # 유사도 상위 3개 청크만 검색
        filter={
            "$and": [
                {"subject": {"$eq": subject}},
                {"unit": {"$eq": unit}},
                {"type": {"$eq": "student_answer"}},
            ]
        },
    )

    # 유사도 점수가 0.3 이하(유사도 높음)인 것만 필터링
    filtered_chunks = [doc for doc, score in results if score <= 0.25]

    if not filtered_chunks:
        print("❗ 유사한 학생 답변 없음")
        return None

    # 추출한 청크 통합
    all_chunk = format_docs(docs=filtered_chunks)

    template = """Evaluate <student answer> based on the <answer key>.

    - Feedback is based on the answer key.
    - Don't evaluate information that's not in the answer key.
    - Focus on conceptual accuracy rather than wording
    - Recognize that foreign proper nouns may be spelled differently.
    - Please write in markdown format and Korean.
    - Organize your feedback under these 2 headings: 
        - ## 누락된 내용 (Missing)
        - ## 틀린 내용 (Incorrect)

    <Answer key>:
    {answer_key}

    <Student answer>:
    {student_answer}
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
    result = rag_chain.invoke(
        {"answer_key": answer_key_chunk, "student_answer": all_chunk}
    )

    # feedback을 ChromaDB에 저장
    vectorstore.add_texts(
        texts=result,
        metadatas=[
            {
                "subject": subject,
                "unit": unit,
                "type": "feedback",
            }
        ],
    )
