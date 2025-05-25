"""
split functions:

1. split text into chunks
2. store chunks into Chroma DB
3. return chunks

"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import re
import os


# ###를 기준으로 청크 나누기 + 상위 제목(#, ##)과 하위 내용을 자동으로 덧붙이기
def split_answer_key(vectorstore: Chroma, subject: str, unit: str, text: str):
    chunks = []
    current_chunk = ""
    h1, h2 = "", ""  # # 대제목, ## 중제목

    lines = text.strip().splitlines()

    for line in lines:
        if line.startswith("# "):
            h1 = line
        elif line.startswith("## "):
            h2 = line
        elif line.startswith("### "):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = ""
            current_chunk += f"{h1}\n{h2}\n{line}\n"
        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    vectorstore.add_texts(
        texts=chunks,
        metadatas=(
            [
                {
                    "subject": subject,
                    "unit": unit,
                    "type": "answer_key",
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]
        ),
    )

    return chunks


# 의미 단위를 만나면 그 문장의 끝에서 split (문장이 끊기지 않게)
def split_student_answer(vectorstore: Chroma, subject: str, unit: str, text: str):
    # 1. 문장 단위로 분리 (마침표, 물음표, 느낌표 포함)
    sentence_endings = re.compile(r"(?<=[.?!])\s+")
    sentences = sentence_endings.split(text.strip())

    chunks = []
    current_chunk = ""

    # 2. 특정 키워드 리스트
    keywords = [
        "첫 번째",
        "두 번째",
        "세 번째",
        "마지막",
        "1.",
        "2.",
        "3.",
        "그러나",
        "하지만",
        "따라서",
        "즉",
        "그래서",
        "예를 들어",
        "결과적으로",
        "정의",
        "개념",
        "특징",
        "이론",
        "공식",
        "설명",
    ]

    for sentence in sentences:
        if current_chunk:
            current_chunk += " " + sentence
        else:
            current_chunk = sentence

        # 문장에 키워드가 있으면 여기서 chunk 분리
        if any(keyword in sentence for keyword in keywords):
            chunks.append(current_chunk.strip())
            current_chunk = ""

        # chunk 크기 제한(선택사항)
        elif len(current_chunk) > 300:
            chunks.append(current_chunk.strip())
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk.strip())

    vectorstore.add_texts(
        texts=chunks,
        metadatas=(
            [
                {
                    "subject": subject,
                    "unit": unit,
                    "type": "student_answer",
                }
                for _ in chunks
            ]
        ),
    )

    return chunks


def format_docs(docs):
    return "\n\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    persist_directory = os.getenv("PERSIST_DIRECTORY")
    dataset_directory = os.getenv("DATASET_DIRECTORY")

    username = "user123"
    answer_key_path = dataset_directory + "/" + "answer_key.txt"
    student_answer_path = dataset_directory + "/" + "student_answer.txt"
    subject = "지구과학"
    unit = "판구조론 정립 과정"

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(),
        collection_name=username,
    )

    with open(answer_key_path, "r", encoding="utf-8") as f:
        docs_answer_key = f.read()
    with open(student_answer_path, "r", encoding="utf-8") as f:
        docs_student_answer = f.read()

    answer_key_chunks = split_answer_key(
        vectorstore=vectorstore, subject=subject, unit=unit, text=docs_answer_key
    )
    student_answer_chunks = split_student_answer(
        vectorstore=vectorstore, subject=subject, unit=unit, text=docs_student_answer
    )

    vectorstore.persist()

    print("\nAnswer_key_chunks:\n")
    for i, chunk in enumerate(answer_key_chunks):
        print(f"\n\n---chunk[{i}]---\n{chunk}")

    print("\Student_answer_chunks:\n")
    for i, chunk in enumerate(student_answer_chunks):
        print(f"\n\n---chunk[{i}]---\n{chunk}")
