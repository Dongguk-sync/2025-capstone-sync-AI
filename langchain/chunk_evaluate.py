import os

from langchain_teddynote import logging

logging.langsmith("Beakji-evaluate")

from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()
persist_directory = os.getenv("PERSIST_DIRECTORY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from split import join_docs
from signup import get_or_create_user_chromadb

SIMILARITY_THRESHOLD = 0.2


def get_evaluation_prompt():
    return """Evaluate <student answer> based on the <answer key>.

    - Feedback is based on the answer key.
    - Don't evaluate information that's not in the answer key.
    - Focus on conceptual accuracy rather than wording
    - Recognize that foreign proper nouns may be spelled differently.
    - Please write in JSON format and Korean.
    - Organize your feedback under these 2 headings: 
        - ## 누락된 내용 (Missing)
        - ## 틀린 내용 (Incorrect)

    <Answer key>:
    {answer_key}

    <Student answer>:
    {student_answer}

    <return example>:
    {{
        "missing": [...],
        "incorrect": [...],
    }}
    """


def chunk_evaluate(vectorstore: Chroma, subject: str, unit: str, answer_key_chunk: str):

    # 벡터 db에서 정답과 관련된 답안 청크 가져오기
    similar_chunks = vectorstore.similarity_search_with_score(
        answer_key_chunk,  # 정답 청크
        k=5,  # 유사도 상위 5개 청크만 검색
        filter={
            "$and": [
                {"subject": {"$eq": subject}},
                {"unit": {"$eq": unit}},
                {"type": {"$eq": "student_answer"}},
            ]
        },
    )

    # 유사도 점수가 SIMILARITY_THRESHOLD 이하(유사도 높음)인 것만 필터링
    filtered_chunks = [
        doc for doc, score in similar_chunks if score <= SIMILARITY_THRESHOLD
    ]

    if not filtered_chunks:
        print("❗ 유사한 학생 답변 없음 (전체 항목 누락)")
        return None

    # 추출한 청크 통합
    all_student_answer_chunk = join_docs(filtered_chunks)

    template = get_evaluation_prompt()

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()

    try:
        result = rag_chain.invoke(
            {"answer_key": answer_key_chunk, "student_answer": all_student_answer_chunk}
        )
    except Exception as e:
        logging.error("🛑 평가 중 오류 발생: %s", e)
        return None

    # feedback을 ChromaDB에 저장
    vectorstore.add_texts(
        texts=[result],
        metadatas=[
            {
                "subject": subject,
                "unit": unit,
                "type": "feedback",
            }
        ],
    )
    return result


if __name__ == "__main__":
    user_id = "user123"
    subject = "지구과학"
    unit = "판구조론 정립 과정"
    index = 0
    answer_key_id = f"{subject}_{unit}_answer_key_{index}"

    vectorstore = get_or_create_user_chromadb(user_id)
    retrieved = vectorstore.get(ids=[answer_key_id])

    if not retrieved["documents"]:
        print(f"❌ ID '{answer_key_id}'에 해당하는 answer key를 찾을 수 없습니다.")
        exit(1)

    answer_key_chunk = retrieved["documents"][0]
    result = chunk_evaluate(vectorstore, subject, unit, answer_key_chunk)
