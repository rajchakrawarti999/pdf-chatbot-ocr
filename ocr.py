import sys
import cv2
import fitz  # PyMuPDF
import numpy as np
import easyocr

MIN_TEXT_LEN = 50  # isse kam text mila to page ko scanned maanenge
_reader = None


def get_reader():
    # EasyOCR tabhi load hoga jab zarurat padegi (digital PDF me nahi)
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en", "hi"], gpu=False)
    return _reader


def preprocess(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.fastNlMeansDenoising(img, None, 10)


def ocr_image(img):
    lines = get_reader().readtext(preprocess(img), detail=0, paragraph=True)
    return "\n".join(lines)


def ocr_pdf_page(page):
    pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return ocr_image(img)


def ocr_file(path, max_pages=None):
    pages = []
    if path.lower().endswith(".pdf"):
        doc = fitz.open(path)
        for i, page in enumerate(doc, start=1):
            if max_pages and i > max_pages:
                break
            text = page.get_text().strip()
            method = "text"
            if len(text) < MIN_TEXT_LEN:
                text = ocr_pdf_page(page)
                method = "ocr"
            pages.append({"page": i, "text": text, "method": method})
        doc.close()
    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image nahi khuli: {path}")
        pages.append({"page": 1, "text": ocr_image(img), "method": "ocr"})
    return pages


if __name__ == "__main__":
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for p in ocr_file(sys.argv[1], max_pages=limit):
        print(f"--- Page {p['page']} ({p['method']}) ---")
        print(p["text"][:300])