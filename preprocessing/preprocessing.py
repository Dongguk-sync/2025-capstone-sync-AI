from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import stt
from dotenv import load_dotenv

import requests
import json
import os

load_dotenv()


class PreprocessingStudentAnswer:
    text = None

    def __init__(self, path):
        self.text = stt.ClovaSpeechClient.req_upload(file=path)

        self.correct_typo()

    def correct_typo(self):
        template = """- Fix typos
- The content is an answer to a test and should not be added to or deleted.
- Write in Korean

<content>:
{content}
"""
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
        corrected_text = rag_chain.invoke({"content": self.text})

        self.text = corrected_text
