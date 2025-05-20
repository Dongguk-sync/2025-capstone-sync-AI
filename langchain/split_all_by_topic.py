# Step 0: Set env

import langchain
import os

langchain.__version__

from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('SECRET_KEY')

# Step 1: Bring an answer key 
# (임시 텍스트 -> DB에서 정답 txt, 답안 txt 받아오기)

docs_answer_key = '''# 판 구조론 정립 과정

## 홈즈의 맨틀 대류설

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

docs_student_answer = '''# 판 구조론 정립 과정

## 대륙 이동설

### 개요
베게너가 주장한 대륙 이동설은 과거 모든 대륙이 모여 판게아라는 초대륙을 이루고 있었으며, 시간이 지나면서 판게아가 분리되고 이동하여 현재와 같은 대륙이 만들어졌다는 이론이다.

### 증거
베게너는 남아메리카 동해안과 아프리카 서해안의 해안선이 일치하는 것과, 서로 떨어진 여러 대륙에서 같은 종의 고생물 화석이 발견되는 것, 북아메리카와 유럽의 산맥 지질 구조가 연속적이고, 여러 대륙의 빙하 흔적을 모으면 빙하의 흔적이 남극을 중심으로 분포하는 것을 대륙 이동설의 증거로 제시하였다.

### 한계
그러나 대륙이 움직이는 원동력을 제대로 설명하지 못해 대다수의 과학자들에게 인정받지 못했다.

## 맨틀 대류설

### 개요
홈스가 주장한 맨틀 대류설은 지구 중심부의 열에 의해 맨틀의 대류가 일어나고, 맨틀 위에 떠 있는 대륙이 이동한다는 이론이다.

### 한계
이 역시 가설을 뒷받침할 수 있는 증거를 제시하지 못해 받아들여지지 않았다.

## 해양저 확장설

### 개요
헤스와 디츠 두 과학자에 의해 발표된 해양저 확장설은 해령 아래에서 맨틀이 상승하여 새로운 해양 지각이 생성되면서 해양저가 확장된다는 이론이다.

### 증거

#### 1. 고지자기 무늬의 대칭적 분포
해령에서 지각이 생성될 때 광물은 당시 자기장의 방향으로 배열되고, 해양 지각이 양옆으로 이동하면서 지구 자기장의 방향이 반대가 되면 새롭게 생성되는 지각의 광물은 반대 방향으로 자화된다.  
이 과정이 반복되면서 해령을 축으로 대칭적인 고지자기 줄무늬가 나타나게 되고, 이것은 해양저 확장설을 지지하는 강력한 증거가 된다.

#### 2. 열곡과 변환 단층의 발견
해양저가 확장하면서 해령의 중심부에 열곡이 만들어지고, 해령의 위치에 따른 속도 차이에 의해 지각이 서로 어긋나는 변환 단층이 생성된다.  
변환 단층에서는 지진이 자주 발생하지만, 해령에서 멀리 떨어진 단열대에서는 지진이 발생하지 않는다.

#### 3. 해양 지각의 나이와 해저 퇴적물의 두께
해령 주변 해양 지각의 나이와 퇴적물의 두께를 측정하였더니, 해령에서 멀어질수록 해양 지각의 나이는 많아지며 퇴적물의 두께는 두꺼워진다는 것을 발견하였고, 이 측정 자료는 해양저 확장설의 증거로 제시되었다.

#### 4. 섭입대 주변의 진원 깊이
해구 부근의 지진은 섭입대를 따라 발생하는데, 대륙 쪽으로 갈수록 진원의 깊이가 깊어진다는 사실을 발견하였고, 이것은 해양에서 생성된 해양 지각이 해구에서 소멸한다는 증거로 활용된다.'''
# docs_student_answer = '''# 판 구조론 정립 과정

# ## 대륙 이동설

# ### 개요
# 베게너가 주장한 대륙 이동설은 과거 모든 대륙이 모여 판게아라는 초대륙을 이루고 있었으며, 시간이 지나면서 판게아가 분리되고 이동하여 현재와 같은 대륙이 만들어졌다는 이론이다.

# ### 증거
# 베게너는 남아메리카 동해안과 아프리카 서해안의 해안선이 일치하는 것과, 서로 떨어진 여러 대륙에서 같은 종의 고생물 화석이 발견되는 것, 북아메리카와 유럽의 산맥 지질 구조가 연속적이고, 여러 대륙의 빙하 흔적을 모으면 빙하의 흔적이 남극을 중심으로 분포하는 것을 대륙 이동설의 증거로 제시하였다.

# ### 한계
# 그러나 대륙이 움직이는 원동력을 제대로 설명하지 못해 대다수의 과학자들에게 인정받지 못했다.

# ## 맨틀 대류설

# ### 개요
# 홈스가 주장한 맨틀 대류설은 지구 중심부의 열에 의해 맨틀의 대류가 일어나고, 맨틀 위에 떠 있는 대륙이 이동한다는 이론이다.

# ### 한계
# 이 역시 가설을 뒷받침할 수 있는 증거를 제시하지 못해 받아들여지지 않았다.

# ## 해양저 확장설

# ### 개요
# 헤스와 디츠 두 과학자에 의해 발표된 해양저 확장설은 해령 아래에서 맨틀이 상승하여 새로운 해양 지각이 생성되면서 해양저가 확장된다는 이론이다.

# ### 증거

# #### 1. 고지자기 무늬의 대칭적 분포
# 해령에서 지각이 생성될 때 광물은 당시 자기장의 방향으로 배열되고, 해양 지각이 양옆으로 이동하면서 지구 자기장의 방향이 반대가 되면 새롭게 생성되는 지각의 광물은 반대 방향으로 자화된다.  
# 이 과정이 반복되면서 해령을 축으로 대칭적인 고지자기 줄무늬가 나타나게 되고, 이것은 해양저 확장설을 지지하는 강력한 증거가 된다.

# #### 2. 열곡과 변환 단층의 발견
# 해양저가 확장하면서 해령의 중심부에 열곡이 만들어지고, 해령의 위치에 따른 속도 차이에 의해 지각이 서로 어긋나는 변환 단층이 생성된다.  
# 변환 단층에서는 지진이 자주 발생하지만, 해령에서 멀리 떨어진 단열대에서는 지진이 발생하지 않는다.

# #### 3. 해양 지각의 나이와 해저 퇴적물의 두께
# 해령 주변 해양 지각의 나이와 퇴적물의 두께를 측정하였더니, 해령에서 멀어질수록 해양 지각의 나이는 많아지며 퇴적물의 두께는 두꺼워진다는 것을 발견하였고, 이 측정 자료는 해양저 확장설의 증거로 제시되었다.

# #### 4. 섭입대 주변의 진원 깊이
# 해구 부근의 지진은 섭입대를 따라 발생하는데, 대륙 쪽으로 갈수록 진원의 깊이가 깊어진다는 사실을 발견하였고, 이것은 해양에서 생성된 해양 지각이 해구에서 소멸한다는 증거로 활용된다.
# '''

print("len(docs_answer_key): " , len(docs_answer_key))
print("len(docs_student_answer): " , len(docs_student_answer))

# Step 2. Split text

# Text Split (Documents -> small chunks: Documents)
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

splits_answer_key = split_by_title(text=docs_answer_key)
splits_student_answer = split_by_title(text=docs_student_answer)

print("len(splits_answer_key):", len(splits_answer_key))
print("len(splits_student_answer):", len(splits_student_answer))

# Step 3: Indexing
# 벡터 DB에 Split된 text를 인덱싱해줌

# Indexing (Texts -> Embedding -> Store)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 벡터 db 생성
vectorstore_student_answer = Chroma.from_texts(texts=splits_student_answer,
                                    embedding=OpenAIEmbeddings())

# 벡터 db에서 특정 내용과 관련된 청크 가져오기
results = vectorstore_student_answer.similarity_search_with_score(
    splits_answer_key[1],   # 정답 청크
    k=1                     # 상위 1개 청크만 검색 => topic 기준으로 split 했으므로 동일 topic 1개 추출
)

# 유사도 점수가 0.3 이하(유사도 높음)인 것만 필터링
filtered_chunks = [doc for doc, score in results if score <= 0.3]

def format_docs(docs):
    return '\n\n\n'.join(doc.page_content for doc in docs)

all_chunk = format_docs(docs=filtered_chunks)

# Step 4: Retrieval ~ Generation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = '''
<Answer Key>에 기반하여 <Student Answer>를 평가해줘.

- 피드백은 <Answer Key>를 기준으로 작성해.
- <Answer Key>에 없는 정보는 평가하지 않아.
- <Answer Key>의 형식을 유지하면서, 빠진 정보는 ==하이라이트== 처리하고, 틀린 내용은 ~~취소선~~으로 표시한 뒤, 수정 내용을 **볼드** 처리해줘.
- 틀린 정보는 반드시 고쳐서 옆에 함께 보여줘.
- 전체 문단 또는 항목을 새로 작성하거나 요약하지 말고, <Answer Key> 구조 그대로 보존한 채 항목 단위로 평가해줘.
- 한국어로 작성해줘.

<Answer key>:
{answer_key}

<Student answer>:
{student_answer}
'''

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model='gpt-4.1-nano-2025-04-14', temperature=0)


rag_chain = (
    RunnablePassthrough()
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain.invoke({
    'answer_key': splits_answer_key[1],
    'student_answer': all_chunk
})
print("\nresult:\n", result)