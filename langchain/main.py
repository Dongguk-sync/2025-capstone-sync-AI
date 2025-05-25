# Step 0: Set env
import langchain
import os

from langchain_teddynote import logging

logging.langsmith("Beakji-evaluate")

from dotenv import load_dotenv

load_dotenv()
persist_directory = os.getenv("PERSIST_DIRECTORY")
dataset_directory = os.getenv("DATASET_DIRECTORY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Step 1: Bring an answer key
# (local 임시 텍스트 -> 나중에 DB에서 정답 txt, 답안 txt 받아오기)

with open(dataset_directory + "/" + "answer_key.txt", "r", encoding="utf-8") as f:
    docs_answer_key = f.read()

with open(dataset_directory + "/" + "student_answer.txt", "r", encoding="utf-8") as f:
    docs_student_answer = f.read()

username = "user123"

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings(),
    collection_name=username,
)
# Step 2: Split text
# Text Split (Documents -> small chunks: Documents)
from split import split_student_answer
from split import split_answer_key
from split import format_docs

splits_answer_key = split_answer_key(text=docs_answer_key)
splits_student_answer = split_student_answer(text=docs_student_answer)

# Step 3: Indexing
# Indexing (Texts -> Embedding -> Store)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Chroma DB 생성 or 불러오기 (collection 없으면 생성, 있으면 불러오기)
vectorstore = Chroma.from_texts(
    texts=splits_answer_key + splits_student_answer,
    embedding=OpenAIEmbeddings(),
    persist_directory=persist_directory,
    collection_name="user123",
    metadatas=(
        [
            {
                "subject": "지구과학",
                "unit": "판구조론의 정립 과정",
                "type": "answer_key",
            }
            for _ in splits_answer_key
        ]
        + [
            {
                "subject": "지구과학",
                "unit": "판구조론의 정립 과정",
                "type": "student_answer",
                "num": "1",
            }
            for _ in splits_student_answer
        ]
    ),
)

# 벡터 db에서 특정 내용과 관련된 청크 가져오기
results = vectorstore.similarity_search_with_score(
    splits_answer_key[1],  # 정답 청크
    k=3,  # 유사도 상위 3개 청크만 검색
    filter={
        "$and": [{"subject": {"$eq": "지구과학"}}, {"type": {"$eq": "student_answer"}}]
    },
)

# 유사도 점수가 0.3 이하(유사도 높음)인 것만 필터링
filtered_chunks = [doc for doc, score in results if score <= 0.25]

# 추출한 청크 통합
all_chunk = format_docs(docs=filtered_chunks)

# Step 4: Retrieval ~ Generation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """Evaluate <student answer> based on the <answer key>.

- Feedback is based on the answer key.
- Don't evaluate information that's not in the answer key.
- Focus on conceptual accuracy rather than wording
- Recognize that foreign proper nouns may be spelled differently.
- Please write in markdown format and Korean.
- Organize your feedback under these 3 headings: 
  - ## 누락된 내용 (Missing)
  - ## 틀린 내용 (Incorrect)
  - ## 요약 평가 (Summary)

<Answer key>:
{answer_key}

<Student answer>:
{student_answer}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()


# 여러 정답 청크에 대해 유사한 학생 답변을 검색하고 평가
results = []
for i, answer_key_chunk in enumerate(splits_answer_key):
    retrieved = vectorstore.similarity_search_with_score(
        answer_key_chunk,
        k=3,
        filter={
            "$and": [
                {"subject": {"$eq": "지구과학"}},
                {"type": {"$eq": "student_answer"}},
            ]
        },
    )
    filtered_chunks = [doc for doc, score in retrieved if score <= 0.25]
    all_chunk = format_docs(docs=filtered_chunks)
    result = rag_chain.invoke(
        {"answer_key": answer_key_chunk, "student_answer": all_chunk}
    )
    results.append(result)

# feedback을 ChromaDB에 저장 (=> 필요한가?)
vectorstore.add_texts(
    texts=result,
    metadatas=[
        {
            "subject": "지구과학",
            "unit": "판구조론의 정립 과정",
            "type": "feedback",
        }
    ]
    * len(results),
)

# local에 영구 저장
vectorstore.persist()
