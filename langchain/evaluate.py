"""
의미 기반 채점 => 정답 + 답안 (전체 text)
"""

import os
from dotenv import load_dotenv

import langchain_chroma
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def initialize_environment():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    logging.langsmith(project_name=os.getenv("LANGSMITH_PROJECT"))

    return {
        "persist_directory": os.getenv("PERSIST_DIRECTORY"),
        "dataset_directory": os.getenv("DATASET_DIRECTORY"),
    }


def get_evaluation_prompt():
    return ChatPromptTemplate.from_template(
        """Evaluate <student answer> based on the <answer key>.

    - Feedback is based on the answer key.
    - Don't evaluate information that's not in the answer key.
    - Focus on conceptual accuracy rather than wording
    - Recognize that foreign proper nouns may be spelled differently.
    - Please write in markdown format and Korean.
    - Organize your feedback under these 3 headings: 
      - # 복습 피드백: {unit}
      - ## 누락 내용 (Missing)
      - ## 틀린 내용 (Incorrect)

    <Answer key>:
    {answer_key}

    <Student answer>:
    {student_answer}
    """
    )


def get_evaluate_result(
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
    model = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0)
    rag_chain = (
        RunnablePassthrough() | get_evaluation_prompt() | model | StrOutputParser()
    )

    result = rag_chain.invoke(
        {
            "unit": unit,
            "answer_key": docs_answer_key,
            "student_answer": docs_student_answer,
        }
    )

    vectorstore.Chroma.add_texts(
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


def load_texts(dataset_directory: str, unit: str):
    answer_key_path = os.path.join(dataset_directory, f"{unit}_answer_key.txt")
    student_answer_path = os.path.join(dataset_directory, f"{unit}_student_answer.txt")

    with open(answer_key_path, "r", encoding="utf-8") as f:
        answer_key = f.read()

    with open(student_answer_path, "r", encoding="utf-8") as f:
        student_answer = f.read()

    return answer_key, student_answer


if __name__ == "__main__":
    from get_chroma import get_or_create_user_chromadb

    env = initialize_environment()
    dataset_dir = env["dataset_directory"]

    user_id = "user123"
    subject = "지구과학"
    unit = "판구조론 정립 과정"

    vectordb = get_or_create_user_chromadb(user_id=user_id)
    answer_key, student_answer = load_texts(dataset_dir, unit)

    feedback = get_evaluate_result(
        vectorstore=vectordb,
        answer_key_text=answer_key,
        student_answer_text=student_answer,
        subject=subject,
        unit=unit,
    )

    print(feedback)
