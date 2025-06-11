"""
회원가입시 Chroma DB의 collection 생성
- collection은 user 당 하나 생성되고, 본인의 collection만 접근 가능
- collection_name이 user_id인 Chroma 반환
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os


def get_or_create_user_chromadb(user_id: str):
    persist_directory = os.getenv("PERSIST_DIRECTORY")
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name=user_id,
        persist_directory=persist_directory,
        embedding_function=embedding,
    )
    return vectorstore


if __name__ == "__main__":
    import chromadb

    get_or_create_user_chromadb(user_id="user123456")

    persist_directory = os.getenv("PERSIST_DIRECTORY")
    client = chromadb.PersistentClient(path=persist_directory)
    collections = client.list_collections()
    print("현재 존재하는 collections:")
    for col in collections:
        print("-", col.name)
