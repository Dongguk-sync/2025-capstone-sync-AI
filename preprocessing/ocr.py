from pdf2image import convert_from_path
import pytesseract
import os


def ocr_from_pdf(pdf_path, dpi=300, lang="eng"):
    images = convert_from_path(
        pdf_path, dpi=dpi, poppler_path="/opt/homebrew/bin"  # poppler가 설치된 경로
    )

    all_text = ""
    for i, image in enumerate(images):
        print(f"[INFO] OCR 처리 중: Page {i+1}/{len(images)}")
        text = pytesseract.image_to_string(image, lang=lang)
        all_text += f"\n\n----- Page {i+1} -----\n{text}"

    return all_text


# 예시 사용
if __name__ == "__main__":
    pdf_path = "../dataset/판구조론 정립 과정 교안.pdf"  # PDF 경로
    output_path = "../dataset/판구조론 정립 과정 교안_extracted_text.txt"

    extracted_text = ocr_from_pdf(pdf_path, lang="kor+eng")  # 한글/영어 혼용일 경우

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"\n✅ 텍스트 추출 완료! → {output_path}")
