import langchain
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("SECRET_KEY")


def markdown_formatting(text):
    template = """- Fix typos
    - Convert the given content to markdown format in korean
    - Use # for the entire article, ## the medium topic, and ### for the small topic.
    - The article is an answer to a test and should not be added to or deleted.

    <content>:
    {content}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
    result = rag_chain.invoke({"content": text})
    return result


if __name__ == "__main__":
    text_path = "../dataset/판구조론 정립 과정 교안_extracted_text.txt"
    output_path = "../dataset/판구조론 정립 과정_answer_key.txt"

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    result = markdown_formatting(text=text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print("포맷팅 성공!\n결과:\n" + result)
