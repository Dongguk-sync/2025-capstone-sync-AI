import os
from langchain_community.document_loaders import UnstructuredPDFLoader

from dotenv import load_dotenv

load_dotenv()


def extract_text_from_pdf(pdf_path: str) -> None:
    """
    PDF 파일에서 텍스트를 추출하고 .txt 파일로 저장하는 함수.
    저장 파일 이름은 입력 파일의 .pdf 확장자를 .txt로 바꾼 이름으로 자동 설정됩니다.

    Args:
        pdf_path (str): 입력 PDF 파일 경로
    """
    # .pdf → .txt 파일명 자동 설정
    base_name = os.path.splitext(pdf_path)[0]
    output_txt_path = f"{base_name}.txt"

    # PDF 로딩 및 텍스트 저장
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents):
            f.write(f"--- 문서 {i+1} ---\n")
            f.write(doc.page_content.strip() + "\n\n")

    print(f"✅ PDF 내용을 '{output_txt_path}' 파일로 저장했습니다.")


if __name__ == "__main__":
    pdf_path = os.getenv("DATASET_DIRECTORY") + "/" + "ch01_데이터베이스 기본 개념.pdf"
    extract_text_from_pdf(pdf_path=pdf_path)
