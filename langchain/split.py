from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import re


# answer_key: 주제(##) 기준으로 정답 청크 나누기
def split_by_title(text):
    chunks = []
    current_chunk = ""
    lines = text.strip().splitlines()

    for line in lines:
        if line.startswith("## ") and current_chunk:  # 새로운 제목을 만나면 청크 분리
            chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            current_chunk += "\n" + line

    if current_chunk:  # 마지막 청크 추가
        chunks.append(current_chunk.strip())

    return chunks


# ###를 기준으로 청크 나누기 + 상위 제목(#, ##)과 하위 내용을 자동으로 덧붙이기
def split_by_subtitle(text):
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

    return chunks


# student_answer: character 기준으로 답안 청크 나누기 (단, 문장의 끝에서 자르기)
def split_by_character(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "? ", "! ", " "],
    )
    chunks = text_splitter.split_text(text)
    return chunks


# 의미 단위 seperator
def split_by_ruled_based(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=[
            "\n\n",
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
            "예를 들어",
            "결과적으로",
            "정의",
            "개념",
            "특징",
            "이론",
            "공식",
            "설명",
            "\n",
            ".",
            "?",
            "!",
            " ",  # fallback
        ],
    )

    chunks = splitter.split_text(text)
    return chunks


# 의미 단위를 만나면 그 문장의 끝에서 split (문장이 끊기지 않게)
def split_by_ruled_sentence(text):
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

    return chunks


# ai에게 chunk의 의미 단위 분석 -> 이건 비용이 많이 들어서 지양
def split_by_hybrid(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "첫 번째", "두 번째", "세 번째", "마지막", ".", " "],
    )
    chunks = splitter.split_text(text)

    prompt = ChatPromptTemplate.from_template(
        """
        다음 텍스트를 읽고, 해당 내용의 주제(제목)와 요약을 작성해주세요.

        ### 텍스트:
        {chunk}

        ### 출력 형식 (JSON):
        {{"title": "...", "summary": "..."}}
        """
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    results = []
    for chunk in chunks:
        messages = prompt.format_messages(chunk=chunk)
        response = llm.invoke(messages)
        results.append(response.content)

    return results


def format_docs(docs):
    return "\n\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    dataset_directory = os.getenv("DATASET_DIRECTORY")

    with open(dataset_directory + "/" + "answer_key.txt", "r", encoding="utf-8") as f:
        docs_answer_key = f.read()

    chunks = split_by_subtitle(docs_answer_key)

    for i, chunk in enumerate(chunks):
        print(f"\n\n---chunk[{i}]---\n{chunk}")
