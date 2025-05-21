import langchain
import os

from dotenv import load_dotenv

load_dotenv()
base_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.getenv("PERSIST_DIRECTORY")
persist_directory = os.path.join(base_dir, relative_path)

os.environ["OPENAI_API_KEY"] = os.getenv("SECRET_KEY")

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

username = "user123"

client = chromadb.PersistentClient(path="../chroma_db")
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings(),
    collection_name=username,
)

results = vectordb.similarity_search(
    query="지각은 어떻게 이동하나요?",
    k=3,
    filter={
        "$and": [{"subject": {"$eq": "지구과학"}}, {"type": {"$eq": "answer_key"}}]
    },
)


def format_docs(docs):
    return "\n\n\n".join(doc.page_content for doc in docs)


print("\n\nquery와 연관된 벡터 검색:\n", format_docs(results))
