# Step 0: Set env
import langchain
import os

from langchain_teddynote import logging

from signup import get_or_create_user_chromadb

logging.langsmith("Beakji-evaluate")

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
persist_directory = os.getenv("PERSIST_DIRECTORY")
dataset_directory = os.getenv("DATASET_DIRECTORY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# answer_key vector 불러오기 -> chunk 순서대로 정렬
def load_answer_key_chunks(vectorstore: Chroma, subject: str, unit: str):
    filtered_docs = vectorstore.get(
        where={"$and": [{"subject": subject}, {"unit": unit}, {"type": "answer_key"}]}
    )

    sorted_docs = sorted(
        zip(filtered_docs["documents"], filtered_docs["ids"]),
        key=lambda x: int(x[1].split("_")[-1]),
    )

    sorted_chunks = [doc for doc, _ in sorted_docs]
    return sorted_chunks


from chunk_evaluate import chunk_evaluate


def evaluate_all_chunks(vectorstore, subject, unit, sorted_chunks):
    all_feedback = []

    for chunk in sorted_chunks:
        result = chunk_evaluate(vectorstore, subject, unit, chunk)
        if result:
            all_feedback.append(result)

    merged_feedback = {"missing": [], "incorrect": []}

    for fb in all_feedback:
        merged_feedback["missing"].extend(fb.get("missing", []))
        merged_feedback["incorrect"].extend(fb.get("incorrect", []))

    merged_feedback["missing"] = list(set(merged_feedback["missing"]))
    merged_feedback["incorrect"] = list(set(merged_feedback["incorrect"]))

    return merged_feedback


if __name__ == "__main__":
    user_id = "user123"
    subject = "지구과학"
    unit = "판구조론 정립 과정"

    vectorstore = get_or_create_user_chromadb(user_id)
    answer_key_chunks = load_answer_key_chunks(vectorstore, subject, unit)

    evaluate_all_chunks(vectorstore, subject, unit, answer_key_chunks)
