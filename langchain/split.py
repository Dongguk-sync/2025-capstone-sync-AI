"""
split 함수:

1. 텍스트를 벡터(작은 덩어리)로 나눔
2. 벡터를 Chroma DB에 저장
3. return 벡터들

"""

import re
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_CHUNK_LENGTH = 300


def _store_chunks(
    vectorstore: Chroma,
    chunks: List[str],
    subject: str,
    unit: str,
    doc_type: str,
    url: str = None,
):
    if not chunks:
        return
    vectorstore.add_texts(
        texts=chunks,
        ids=[f"{subject}_{unit}_{doc_type}_{i}" for i in range(len(chunks))],
        metadatas=[
            {
                "subject": subject,
                "unit": unit,
                "type": doc_type,
                **({"url": url} if url else {}),
            }
            for _ in chunks
        ],
    )


def split_answer_key(
    vectorstore: Chroma, subject: str, unit: str, text: str, url: str
) -> List[str]:
    chunks = []
    current_chunk = ""
    h1, h2 = "", ""
    for line in text.strip().splitlines():
        if line.startswith("# "):
            h1 = line
        elif line.startswith("## "):
            h2 = line
        elif line.startswith("### "):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = f"{h1}\n{h2}\n{line}\n"
        else:
            current_chunk += line + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    _store_chunks(vectorstore, chunks, subject, unit, "answer_key", url)
    return chunks


def split_student_answer(
    vectorstore: Chroma,
    subject: str,
    unit: str,
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 100,
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "? ", "! ", " "],
    )
    chunks = splitter.split_text(text)
    _store_chunks(vectorstore, chunks, subject, unit, "student_answer")
    return chunks
