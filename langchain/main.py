# Step 0: Set env
import langchain
import os

from langchain_teddynote import logging

logging.langsmith("Beakji-evaluate")

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
persist_directory = os.getenv("PERSIST_DIRECTORY")
dataset_directory = os.getenv("DATASET_DIRECTORY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

with open(dataset_directory + "/" + "answer_key.txt", "r", encoding="utf-8") as f:
    docs_answer_key = f.read()

with open(dataset_directory + "/" + "student_answer.txt", "r", encoding="utf-8") as f:
    docs_student_answer = f.read()

username = "user123"

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings(),
    collection_name=username,
)

# 여러 정답 청크에 대해 유사한 학생 답변을 검색하고 평가
results = []
for i, answer_key_chunk in enumerate(splits_answer_key):

    results.append(result)

# local에 영구 저장
vectorstore.persist()
