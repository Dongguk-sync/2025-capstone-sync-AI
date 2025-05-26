import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

dataset_directory = os.getenv("DATASET_DIRECTORY")


def correct_typo(text: str):
    template = """- Fix typos
    - The content is an answer to a test and should not be added to or deleted.
    - Write in Korean

    <content>:
    {content}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
    result = rag_chain.invoke({"content": text})
    return result


if __name__ == "__main__":
    unit = "판구조론 정립 과정"
    source_path = dataset_directory + "/" + "student_answer.txt"
    student_answer_path = dataset_directory + "/" + unit + "_student_answer.txt"
    with open(source_path, "r", encoding="utf-8") as f:
        docs_student_answer = f.read()
    result = correct_typo(docs_student_answer)
    with open(student_answer_path, "w", encoding="utf-8") as f:
        f.write(result)
