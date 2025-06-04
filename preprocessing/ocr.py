import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from fastapi import APIRouter, UploadFile, File
from services.ocr_service import process_pdf

router = APIRouter()
load_dotenv()

poppler_path = os.getenv("POPPLER_PATH")


def ocr_from_pdf(pdf_path, dpi=300, lang="kor+eng") -> None:

    base_name = os.path.splitext(pdf_path)[0]
    output_path = f"{base_name}.txt"

    images = convert_from_path(
        pdf_path, dpi=dpi, poppler_path=poppler_path  # poppler가 설치된 경로
    )

    all_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang=lang)
        all_text += f"\n{text}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_text)


if __name__ == "__main__":
    pdf_path = os.getenv("DATASET_DIRECTORY") + "/" + "ch01_데이터베이스 기본 개념.pdf"
    ocr_from_pdf(pdf_path, lang="kor")  # 한글로만 구성 -> 정확성
