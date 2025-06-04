import requests
import json
import os
from dotenv import load_dotenv

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


if __name__ == "__main__":
    client = ClovaSpeechClient()

    file_path = "/Users/kimsuyoung/Desktop/대학/25-1/캡스톤디자인1/test/판구조론 정립과정(해양저 확장설) [dnYUwb7j5Xc].mp3"

    result = client.req_upload(
        file_path=file_path,
        completion="sync",
        diarization={"enable": False},
    )

    print(result)
