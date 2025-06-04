"""
학생의 복습 음성 전처리

1. STT 수행 (Clova Speech API)
2. 맞춤법 교정 (LLM)
3. 결과 저장 및 반환
"""

import os
import requests
import json
from dotenv import load_dotenv
from preprocessing import correct_typo

load_dotenv()


class ClovaSpeechClient:
    def __init__(self):
        self.invoke_url = os.getenv("CLOVA_INVOKE_URL")
        self.secret = os.getenv("CLOVA_API_KEY")

    def req_url(
        self,
        url,
        completion,
        callback=None,
        userdata=None,
        forbiddens=None,
        boostings=None,
        wordAlignment=True,
        fullText=True,
        diarization=None,
        sed=None,
    ):
        request_body = {
            "url": url,
            "language": "ko-KR",
            "completion": completion,
            "callback": callback,
            "userdata": userdata,
            "wordAlignment": wordAlignment,
            "fullText": fullText,
            "forbiddens": forbiddens,
            "boostings": boostings,
            "diarization": diarization,
            "sed": sed,
        }
        headers = {
            "Accept": "application/json;UTF-8",
            "Content-Type": "application/json;UTF-8",
            "X-CLOVASPEECH-API-KEY": self.secret,
        }
        return requests.post(
            headers=headers,
            url=self.invoke_url + "/recognizer/url",
            data=json.dumps(request_body).encode("UTF-8"),
        )

    def req_upload(
        self,
        file,
        completion="sync",
        callback=None,
        userdata=None,
        forbiddens=None,
        boostings=None,
        wordAlignment=True,
        fullText=True,
        diarization={"enable": False},
        sed=None,
    ):
        request_body = {
            "language": "ko-KR",
            "completion": completion,
            "callback": callback,
            "userdata": userdata,
            "wordAlignment": wordAlignment,
            "fullText": fullText,
            "forbiddens": forbiddens,
            "boostings": boostings,
            "diarization": diarization,
            "sed": sed,
        }
        headers = {
            "Accept": "application/json;UTF-8",
            "X-CLOVASPEECH-API-KEY": self.secret,
        }
        print(json.dumps(request_body, ensure_ascii=False).encode("UTF-8"))
        files = {
            "media": open(file, "rb"),
            "params": (
                None,
                json.dumps(request_body, ensure_ascii=False).encode("UTF-8"),
                "application/json",
            ),
        }
        response = requests.post(
            headers=headers, url=self.invoke_url + "/recognizer/upload", files=files
        )
        return response.text


def preprocess_student_answer(
    audio_file_path: str,
    unit: str,
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

    output_path = os.path.join(dataset_directory, f"{unit}_student_answer.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corrected_text)

    return corrected_text


if __name__ == "__main__":
    client = ClovaSpeechClient()

    file_path = "/Users/kimsuyoung/Desktop/대학/25-1/캡스톤디자인1/test/판구조론 정립과정(해양저 확장설) [dnYUwb7j5Xc].mp3"

    result = client.req_upload(
        file_path=file_path,
        completion="sync",
        diarization={"enable": False},
    )

    print(result)
