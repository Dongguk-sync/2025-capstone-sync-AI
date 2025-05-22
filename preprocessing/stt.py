import requests
import json


class ClovaSpeechClient:
    # Clova Speech invoke URL
    invoke_url = 'https://clovaspeech-gw.ncloud.com/external/v1/10986/a59163b9fbfc040f8df27292575b44752397b87a9353133f5021ab654b10113b'
    # Clova Speech secret key
    secret = 'bbe8ff2a229c42e48b1b1b442c8bd985'

    def req_url(self, url, completion, callback=None, userdata=None,
                forbiddens=None, boostings=None, wordAlignment=True,
                fullText=True, diarization=None, sed=None):
        request_body = {
            'url': url,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/url',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_object_storage(self, data_key, completion, callback=None,
                           userdata=None, forbiddens=None, boostings=None, wordAlignment=True,
                           fullText=True, diarization=None, sed=None):
        request_body = {
            'dataKey': data_key,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/object-storage',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_upload(self, file, completion, callback=None, userdata=None,
                   forbiddens=None, boostings=None, wordAlignment=True,
                   fullText=True, diarization=None, sed=None):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        print(json.dumps(request_body, ensure_ascii=False).encode('UTF-8'))
        files = {
            'media': open(file, 'rb'),
            'params': (None, json.dumps(request_body,
                                        ensure_ascii=False).encode('UTF-8'),
                       'application/json')
        }
        response = requests.post(headers=headers, url=self.invoke_url
                                 + '/recognizer/upload', files=files)
        return response


if __name__ == '__main__':
    res = ClovaSpeechClient().req_upload(
        file='/Users/kimsuyoung/Desktop/대학/25-1/캡스톤디자인1/test/판구조론 정립과정(해양저 확장설) [dnYUwb7j5Xc].mp3', completion='sync', diarization={
            "enable": False
        })
    print(res.text)
