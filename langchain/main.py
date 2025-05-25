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


# answer_key vector 불러오기 -> chunk 순서대로 정렬
def load_answer_key_chunks(collection, doc_id):
    result = collection.get(where={"doc_id": doc_id})
    return sorted(
        zip(result["metadatas"], result["documents"]),
        key=lambda x: x[0]["chunk_index"],
    )


from chunk_evaluate import chunk_evaluate


def evaluate_all_chunks(vectorstore, domain, topic, sorted_chunks):
    return [
        chunk_evaluate(vectorstore, domain, topic, chunk) for chunk in sorted_chunks
    ]
