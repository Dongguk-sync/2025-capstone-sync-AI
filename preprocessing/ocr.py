import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


class Ocr:
    def __init__(self):
        self.poppler_path = os.getenv("POPPLER_PATH")

    def ocr_from_pdf(self, pdf_path, dpi=300, lang="kor+eng") -> str:

        images = convert_from_path(
            pdf_path, dpi=dpi, poppler_path=self.poppler_path  # poppler가 설치된 경로
        )

        all_text = ""
        for _, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang=lang)
            all_text += f"\n{text}"

        return all_text


def markdown_formatting(text) -> str:
    template = """- Convert the given content to markdown format in korean
    - The given text is from a textbook. Extract only the concepts, not the exercises in the textbook.
    - Use # for the entire article, ## the medium topic, and ### for the small topic.
    - The article is an answer key to a test and should not be added to or deleted.

    <content>:
    {content}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
    result = rag_chain.invoke({"content": text})
    return result


def preprocess_answer_key(pdf_path: str) -> str:
    ocr = Ocr()
    raw_text = ocr.ocr_from_pdf(pdf_path)
    markdown_text = markdown_formatting(raw_text)
    return markdown_text


if __name__ == "__main__":
    dataset_dir = os.getenv("DATASET_DIRECTORY")
    pdf_path = os.path.join(dataset_dir, "ch01_데이터베이스 기본 개념.pdf")

    result = preprocess_answer_key(pdf_path)
    print(result)
