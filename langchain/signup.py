"""
회원가입시 Chroma DB의 collection 생성
collection은 user 당 하나 생성되고, 본인의 collection만 접근 가능
"""

from chromadb import Client
from chromadb.config import Settings
import os


def signup(user_id: str) -> None:
    persist_directory = os.getenv("PERSIST_DIRECTORY")
    client = Client(Settings(persist_directory=persist_directory))

    # Create a collection named after the user_id if it doesn't already exist
    collection = client.get_or_create_collection(name=user_id)

    print(f"Collection for user '{user_id}' created or already exists.")
