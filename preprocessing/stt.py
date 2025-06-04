"""
학생의 복습 음성 전처리

1. STT 수행 (Clova Speech API)
2. 맞춤법 교정 (LLM)
3. 결과 저장 및 반환
"""

import sys
import os
import requests
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from preprocessing.split import split_student_answer

load_dotenv()


class ClovaSpeechClient:
    def __init__(self):
        self.invoke_url = os.getenv("CLOVA_INVOKE_URL")
        self.secret = os.getenv("CLOVA_API_KEY")

    def req_upload(self, file, completion="sync") -> str:
        request_body = {
            "language": "ko-KR",
            "completion": completion,
            "wordAlignment": True,
            "fullText": True,
            "diarization": {"enable": False},
        }
        headers = {
            "Accept": "application/json;UTF-8",
            "X-CLOVASPEECH-API-KEY": self.secret,
        }
        files = {
            "media": open(file, "rb"),
            "params": (
                None,
                json.dumps(request_body, ensure_ascii=False).encode("UTF-8"),
                "application/json",
            ),
        }
        response = requests.post(
            url=self.invoke_url + "/recognizer/upload", headers=headers, files=files
        )
        return response.text


def correct_typo(text: str) -> str:
    template = """- Fix typos
    - The content is an answer to a test and should not be added to or deleted.
    - Write in Korean

    <content>:
    {content}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
    result = rag_chain.invoke({"content": text})
    return result


def preprocess_student_answer(
    audio_file_path: str,
    unit: str,
    subject: str,
    username: str,
    dataset_directory: str,
    clova_client: ClovaSpeechClient,
) -> str:
    stt_result_json = clova_client.req_upload(
        file=audio_file_path,
        completion="sync",
        diarization={"enable": False},
    )

    try:
        stt_text = json.loads(stt_result_json)["text"]
    except Exception as e:
        raise ValueError(f"STT 결과 파싱 실패: {e}")

    corrected_text = correct_typo(stt_text)

    # 텍스트 저장
    output_path = os.path.join(dataset_directory, f"{unit}_student_answer.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corrected_text)

    # 벡터 저장
    vectorstore = get_or_create_user_chromadb(username)
    split_student_answer(vectorstore, subject, unit, corrected_text)

    return corrected_text


if __name__ == "__main__":
    client = ClovaSpeechClient()

    # 예시 값 (테스트 시 실제 값으로 변경)
    file_path = "/Users/kimsuyoung/Desktop/대학/25-1/캡스톤디자인1/test/판구조론 정립과정(해양저 확장설) [dnYUwb7j5Xc].mp3"
    dataset_dir = "./dataset"
    subject = "지구과학"
    unit = "판구조론"
    username = "kimsuyoung"

    # 전처리 및 벡터 저장까지 실행
    vectorstore = get_or_create_user_chromadb(username)
    text = preprocess_student_answer(
        audio_file_path=file_path,
        unit=unit,
        subject=subject,
        username=username,
        dataset_directory=dataset_dir,
        clova_client=client,
    )
