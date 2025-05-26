"""
prompt: 정답 + 답안 (단순 비교)
"""

# Step 0: Set env
import langchain
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_teddynote import logging

logging.langsmith(project_name=os.getenv("LANGSMITH_PROJECT"))

persist_directory = os.getenv("PERSIST_DIRECTORY")

# 정답, 답안 파일 경로 지정
answer_key_path = os.getenv("DATASET_DIRECTORY") + "/" + "answer_key.txt"
student_answer_path = os.getenv("DATASET_DIRECTORY") + "/" + "student_answer.txt"

# Step 1: Bring text files
# (local 임시 텍스트 -> 나중에 DB에서 정답 txt, 답안 txt 받아오기)

with open(answer_key_path, "r", encoding="utf-8") as f:
    docs_answer_key = f.read()

with open(student_answer_path, "r", encoding="utf-8") as f:
    docs_student_answer = f.read()

# Step 2: Retrieval ~ Generation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """Evaluate <student answer> based on the <answer key>.

- Feedback is based on the answer key.
- Don't evaluate information that's not in the answer key.
- Focus on conceptual accuracy rather than wording
- Recognize that foreign proper nouns may be spelled differently.
- Please write in markdown format and Korean.
- Organize your feedback under these 3 headings: 
  - ## 누락된 내용 (Missing)
  - ## 틀린 내용 (Incorrect)
  - ## 요약 평가 (Summary)

<Answer key>:
{answer_key}

<Student answer>:
{student_answer}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0)

rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()

result = rag_chain.invoke(
    {"answer_key": docs_answer_key, "student_answer": docs_student_answer}
)
print("\nanswer_key:\n", docs_answer_key)
print("\nstudent_answer:\n", docs_student_answer)
print("\nresult:\n", result)
