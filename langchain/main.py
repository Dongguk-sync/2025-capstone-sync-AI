"""
교안 PDF를 입력 받았을 시:
OCR -> markdown_formatting -> ChromaDB 저장 (answer_key)

복습 음성을 입력 받았을 시:
STT -> correct_typo -> ChromaDB 저장 (student_answer)

채점시:
answer_key, student_answer -> compare_evaluate -> ChromaDB 저장 (Feedback)
"""

import langchain
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote import logging
from dotenv import load_dotenv
from chunk_evaluate import chunk_evaluate
from signup import get_or_create_user_chromadb

logging.langsmith("Beakji-evaluate")
load_dotenv()
persist_directory = os.getenv("PERSIST_DIRECTORY")
dataset_directory = os.getenv("DATASET_DIRECTORY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
