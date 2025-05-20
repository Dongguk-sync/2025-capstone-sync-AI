# Step 0. Set env

import langchain
import os

langchain.__version__

from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('SECRET_KEY')

# Step 1. Bring an answer key 
# (임시 텍스트 -> DB에서 받아오기)

docs_answer_key = '''## 홈즈의 맨틀 대류설
### 맨틀 대류설의 등장 배경
- 베게너의 대륙 이동설이 받아들여지지 않았던 이유 중 하나는 대륙 이동의 원동력에 대한 설명 부족
- 아서 홈즈가 대안 제시

### 홈즈의 주장
- 1929년, 대륙 이동의 원동력으로 ‘맨틀의 대류’를 제안
- 지각 아래의 맨틀에서 온도 차이에 의해 대류 발생 → 지각 이동
- 대류 속도: 연간 몇 cm 수준 (매우 느림)
- 대류 원인: 지구 중심부의 높은 온도

### 한계와 의의
- 대류 깊이, 크기 등은 명확히 설명하지 못함
- 그러나 판구조론의 핵심 요소 포함
  - 대륙 지각과 해양 지각의 이동
  - 맨틀 대류 하강 → 해구 생성
  - 맨틀 대류 상승 → 해령 생성
- 당시는 실험/관찰로 증명 어려워 인정받지 못함

### 업적 기념
- 유럽지질과학협회: 뛰어난 지질학자에게 ‘아서 홈즈 메달’ 수여

---

## 헤스와 디츠의 해저 확장설
### 해저 지형 탐사
- 1920년대: 음파 이용한 해저 지형 조사 시작
- 1947년: 대서양 중앙 해저 산맥 발견
- 1953년: 산맥 중심의 열곡대 발견

### 해저 확장설 개요
- 발표: 1960년 헤리 H. 헤스, 1961년 로버트 S. 디츠
- 내용:
  - 해양 지각은 해령에서 생성
  - 맨틀 대류 따라 이동 후 해구에서 소멸
  - 마그마가 열곡에서 분출 → 식어서 새로운 해양 지각 형성

### 증거
- 해령에서 해구로 갈수록
  - 지각의 나이 증가
  - 퇴적물 두께 증가
  - 수심 증가
- 대륙 이동설과 맨틀 대류설을 뒷받침
- 윌슨의 판구조론 연구에 기여

---

## 윌슨의 변환 단층
### 윌슨의 업적
- 캐나다 지질학자
- 대륙 이동설 + 해저 확장설 통합하여 판구조론 정립 시도

### 변환 단층 개념
- 1965년, 서로 반대 방향으로 미끄러지는 지각 발견 → ‘변환 단층’ 명명
- 해구, 해령과 달리 지각이 새로 만들어지거나 소멸되지 않음
- 판의 수평 이동이 주된 특징

### 특징
- 주로 대양저 산맥 주변에서 발생
- 해구나 육지에서도 나타남
- 천발 지진 발생

---

## 판구조론의 탄생
### 학술적 발전
- 대륙 이동설, 맨틀 대류설, 해저 확장설에 대한 관심 증대
- 1967년 미국 지구물리연맹 학술회의: 해저 확장설 논문 70편 발표

### 모건의 연구 (1968년 3월)
- 지진 발생 지역이 판의 경계임을 설명
- 판의 두께: 약 100km
- 6개의 대형 판 + 12개의 소형 판 구분
- 운동 방향과 상대 속도 계산

### 지진학적 증거 (1968년 9월)
- 아이악스, 올리버, 사이크스: 판 운동 증명
- ‘tecton(건축자)’에서 유래된 ‘tectonis’ 용어 도입
- 최초로 ‘신지구 구조론(New Global Tectonics)’ 사용
- 이후 ‘판구조론(plate tectonics)’으로 불림
'''

docs_student_answer = '''

판 구조론 정립 과정

판 구조론은 대륙 이동설, 멘틀 대류설, 해양저 확장설을 거치면서 완성되었다.

⸻

1. 대륙 이동설 (베게너)
	•	과거 모든 대륙이 판게아라는 초대륙을 이루었으며, 시간이 지나면서 판게아가 분리되고 이동하여 현재와 같은 대륙이 형성되었다는 이론이다.
	•	주요 증거:
	•	남아메리카 동해안과 아프리카 서해안의 해안선 일치
	•	서로 떨어진 여러 대륙에서 같은 종의 고생물 화석 발견
	•	북아메리카와 유럽의 산맥 지질 구조가 연속적
	•	여러 대륙의 빙하 흔적을 모으면 남극 중심으로 분포

→ 하지만 대륙 이동의 원동력을 설명하지 못해 당시 과학계에서 인정받지 못함.

⸻

2. 멘틀 대류설 (홈스)
	•	지구 중심부의 열에 의해 멘틀 대류가 발생하고, 멘틀 위에 떠 있는 대륙이 이동한다는 이론.
	•	→ 그러나 가설을 뒷받침할 증거 부족으로 받아들여지지 않음.

⸻

3. 해양저 확장설 (헤스와 디츠)
	•	해령 아래에서 멘틀이 상승하여 새로운 해양 지각이 생성, 해양저가 양옆으로 확장된다는 이론.

주요 증거:
	1.	고지자기 무늬의 대칭적 분포
	•	해령에서 생성된 광물은 당시 자기장의 방향으로 배열됨.
	•	자기장이 반전되면, 새롭게 생성되는 지각의 광물은 반대 방향으로 자화됨.
	•	이 과정이 반복되며, 해령을 중심으로 대칭적인 고지자기 줄무늬 형성 → 해양저 확장설의 강력한 증거.
	2.	열곡과 변환 단층의 존재
	•	해양저가 확장하면서 해령 중심부에 열곡이 형성됨.
	•	해령의 위치에 따른 속도 차이로 지각이 어긋나며 변환 단층 생성.
	•	변환 단층에서는 지진이 자주 발생, 해령에서 멀리 떨어진 단열대에서는 지진이 발생하지 않음.
	3.	해양 지각의 나이와 해저 퇴적물 두께
	•	해령에서 멀어질수록:
	•	해양 지각의 나이 증가
	•	퇴적물 두께 증가
	•	→ 이 측정 자료도 해양저 확장설의 증거가 됨.
	4.	섭입대 주변의 진원 깊이 분포
	•	해구 근처 지진은 섭입대를 따라 발생.
	•	대륙 쪽으로 갈수록 진원의 깊이가 깊어짐.
	•	→ 해양에서 생성된 지각이 해구에서 소멸된다는 증거로 활용됨.
'''

print("len(docs_answer_key): " , len(docs_answer_key))
# print(len(docs_answer_key[0].page_content))
# print(docs_answer_key[0].page_content[100:200])

# Step 2. Split text

# Text Split (Documents -> small chunks: Documents)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 덩어리 사이즈 지정, 겹치는 영역 사이즈 지정
# 문장 단위로 덩어리 쪼갬
text_splitter_answer_key = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
text_splitter_student_answer = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)

# 문장의 끝까지 나눠줌. 문장이 중간에 끊겨 불러와지지 않도록 함
splits_answer_key = text_splitter_answer_key.split_text(docs_answer_key)
splits_student_answer = text_splitter_student_answer.split_text(docs_student_answer)

print("len(splits_answer_key)", len(splits_answer_key))
print("splits_answer_key[1]", splits_answer_key[1])

# page_content 속성
# splits_answer_key[10].page_content

# metadata 속성
# splits_answer_key[10].metadata

# Step 3: Indexing
# 벡터 DB에 Split된 text를 인덱싱해줌

# Indexing (Texts -> Embedding -> Store)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 벡터 db 생성

# vectorstore_answer_key = Chroma.from_documents(documents=splits_answer_key,
#                                     embedding=OpenAIEmbeddings())
# vectorstore_student_answer = Chroma.from_documents(documents=splits_student_answer,
#                                     embedding=OpenAIEmbeddings())

vectorstore_answer_key = Chroma.from_texts(texts=splits_answer_key, embedding=OpenAIEmbeddings())
vectorstore_student_answer = Chroma.from_texts(texts=splits_student_answer,
                                    embedding=OpenAIEmbeddings())

# 벡터 db에서 특정 내용과 관련된 청크 가져오기

# docs_answer_key = vectorstore_answer_key.similarity_search("판 구조론 정립과정")
# print(len(docs_answer_key))
# print(docs_answer_key[0].page_content)

chunk = vectorstore_student_answer.similarity_search(splits_answer_key[0])
print(len(chunk))
print(chunk[0].page_content)

# Step 4: Retrieval ~ Generation
# 벡터 DB에서 유사한 내용을 불러와 prompt에 포함시키기

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = '''
Provide feedback of <Student answer> based on <Answer key> that distinguishes between errors and omissions.
Format in Markdown and in Korean.


<Answer key>: {context}

<Student answer>: {input}
'''
# prompt = ChatPromptTemplate.from_template(template)

# retriever = vectorstore_answer_key.as_retriever()

# docs_answer_key = vectorstore_answer_key.similarity_search("{input}")

# def format_docs(docs):
#     return '\n\n'.join(doc.page_content for doc in docs)

# rag_chain = (
#     {'context': retriever | format_docs, 'input': RunnablePassthrough()}
#     | prompt
#     | model
#     | StrOutputParser()
# )
# review = '''이번 수업은 판 구조론 정립 과정에 대해 정리해 보자. 판 구조로는 대륙 이동설, 멘틀 대류설, 해양저 확장설을 거치면서 완성되었다. 베게너가 주장한 대륙 이동설은 과거 모든 대륙이 모여 판게아라는 초대륙을 이루고 있었으며, 시간이 지나면서 판게아가 분리되고 이동하여 현재와 같은 대륙이 만들어졌다는 이론이다. 베게너는 남아메리카 동해안과 아프리카 서해안의 해안선이 일치하는 것과 서로 떨어진 여러 대륙에서 같은 종의 고생물 화석이 발견되는 것. 북아메리카와 유럽의 산맥 지질 구조가 연속적이고 여러 대륙의 빙하 흔적을 모으면 빙하의 흔적이 남극을 중심으로 분포하는 것을 대륙 이동설의 증거로 제시하였지만, 대륙이 움직이는 원동력을 제대로 설명하지 못해 대다수의 과학자들에게 인정받지 못했다. 홈스가 주장한 멘틀 대류설은 지구 중심부의 열에 의해 멘틀의 대류가 일어나고 멘틀 위에 떠 있는 대륙이 이동한다는 이론으로 이 역시 가설을 뒷받침할 수 있는 증거를 제시하지 못해 받아들여지지 않았다. 헤스와 디츠 두 과학자에 의해 발표된 해양저 확장설은 해령 아래에서 멘틀이 상승하여 새로운 해양 지각이 생성되면서 해양자가 확장된다는 이론이다. 해양저 확장설의 여러 증거들을 살펴보자. 첫 번째 증거는 고지자기 무늬의 대칭적 분포이다. 해량에서 지각이 생성될 때 광물은 당시 자기장의 방향으로 배열되고 해양 지각이 양옆으로 이동하면서 지구 자기장의 방향이 반대가 되면 새롭게 생성되는 지각의 광물은 반대 방향으로 좌화된다. 이 과정이 반복되면서 해령을 축으로 대칭적인 고지자기 줄무늬가 나타나게 되고, 이것은 해양전 확장설을 지지하는 강력한 증거가 된다. 두 번째 증거는 열곡과 변환 단층이 발견된다는 사실이다. 해양자가 확장하면서 해령의 중심부에 열곡이 만들어지고 해령의 위치에 따른 속도 차이에 의해 지각이 서로 어긋나는 변환 단층이 생성된다. 변환 단층에서는 지진이 자주 발생하지만 해령에서 멀리 떨어진 단열대에서는 지진이 발생하지 않는다. 세 번째 증거는 해양 지각의 나이와 해저 퇴적물의 두께이다. 해령 주변 해양 지각의 나이와 퇴적물의 두께를 측정하였더니 해령에서 멀어질수록 해양 지각의 나이는 많아지며 퇴적물의 두께는 두꺼워진다는 것을 발견하였고, 이 측정 자료는 해양저 확장설의 증거로 제시되었다. 마지막 증거는 섭입대 주변의 진원 깊이이다. 해구 부근의 지진은 섭입대를 따라 발생하는데 대륙 쪽으로 갈수록 진원의 깊이가 깊어진다는 사실을 발견하였고, 이것은 해양에서 생성된 해양 지각이 해구에서 소멸한다는 증거로 활용된다.'''
# rag_chain.invoke(review)


template = '''
Provide feedback of <Student answer> based on <Answer key> that distinguishes between errors and omissions.
Format in Markdown and in Korean.


<Answer key>: {answer_key}

<Student answer>: {student_answer}
'''
prompt = ChatPromptTemplate.from_template(template)

retriever = vectorstore_answer_key.as_retriever()

docs = vectorstore_answer_key.similarity_search("{student_answer}")

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

model = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)

rag_chain = (
    {'answer_key': retriever | format_docs, 'student_answer': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
review = splits_answer_key[0] # or vectorstore_answer_key[0]
result = rag_chain.invoke(review)

print("\n다음 내용을 채점합니다(answer_key):\n", splits_answer_key[0])
print("\n학생의 관련된 응답입니다:\n", format_docs(docs=docs))
print("\nresult:\n", result)