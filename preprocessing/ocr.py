from pdf2image import convert_from_path
import pytesseract


def ocr_from_pdf(pdf_path, dpi=300, lang="kor+eng"):  # 한글과 영어가 혼용된 경우
    images = convert_from_path(
        pdf_path, dpi=dpi, poppler_path="/opt/homebrew/bin"  # poppler가 설치된 경로
    )

    all_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang=lang)
        all_text += f"\n{text}"
    return all_text


if __name__ == "__main__":
    pdf_path = "../dataset/판구조론 정립 과정 교안.pdf"  # PDF 경로
    output_path = "../dataset/판구조론 정립 과정 교안_extracted_text.txt"

    extracted_text = ocr_from_pdf(pdf_path, lang="kor")  # 한글로만 구성 -> 정확성

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"\n텍스트 추출 성공\n결과:\n{extracted_text}")
