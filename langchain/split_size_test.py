from langchain.text_splitter import RecursiveCharacterTextSplitter


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


# student_answer: character 기준으로 답안 청크 나누기 (단, 문장의 끝에서 자르기)
def split_by_character(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "? ", "! ", " "],
    )
    chunks = text_splitter.split_text(text)
    return chunks


def format_docs(docs):
    return "\n\n\n".join(doc.page_content for doc in docs)
