"""
prompt: 정답 + 답안 (단순 비교)
"""

# Step 0: Set env
import langchain
import os

from dotenv import load_dotenv
import langchain_chroma

from langchain_teddynote import logging

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

logging.langsmith(project_name=os.getenv("LANGSMITH_PROJECT"))

persist_directory = os.getenv("PERSIST_DIRECTORY")
dataset_directory = os.getenv("DATASET_DIRECTORY")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_evaluation_prompt():
    return """Evaluate <student answer> based on the <answer key>.

    - Feedback is based on the answer key.
    - Don't evaluate information that's not in the answer key.
    - Focus on conceptual accuracy rather than wording
    - Recognize that foreign proper nouns may be spelled differently.
    - Please write in markdown format and Korean.
    - Organize your feedback under these 3 headings: 
      - # {unit} Feedback
      - ## 누락된 내용 (Missing)
      - ## 틀린 내용 (Incorrect)

    <Answer key>:
    {answer_key}

    <Student answer>:
    {student_answer}
    """


def compare_evaluate(
    vectorstore: langchain_chroma,
    answer_key_path: str,
    student_answer_path: str,
    subject: str,
    unit: str,
) -> str:
    # Step 1: Bring text files
    with open(answer_key_path, "r", encoding="utf-8") as f:
        docs_answer_key = f.read()

    with open(student_answer_path, "r", encoding="utf-8") as f:
        docs_student_answer = f.read()

    # Step 2: Retrieval ~ Generation
    prompt = ChatPromptTemplate.from_template(get_evaluation_prompt())
    model = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0)
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()

    result = rag_chain.invoke(
        {
            "unit": unit,
            "answer_key": docs_answer_key,
            "student_answer": docs_student_answer,
        }
    )

    vectorstore.add_texts(
        texts=[result],
        metadatas=[
            {
                "subject": subject,
                "unit": unit,
                "type": "feedback",
            }
        ],
    )
    return result


if __name__ == "__main__":
    from signup import get_or_create_user_chromadb

    user_id = "user123"
    subject = "지구과학"
    unit = "판구조론 정립 과정"

    vectorstore = get_or_create_user_chromadb(user_id=user_id)

    answer_key_path = dataset_directory + "/" + unit + "_answer_key.txt"
    student_answer_path = dataset_directory + "/" + unit + "_student_answer.txt"

    compare_evaluate(vectorstore, answer_key_path, student_answer_path, subject, unit)
