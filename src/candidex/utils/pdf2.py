from pathlib import Path

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams


def extract_text_pdfminer_high(pdf_path: Path) -> str:
    """Extracts text from a digital PDF using pdfminer.six high-level
    API."""
    # LAParams lets you fine-tune how blocks, words, and lines are detected.
    # Leaving it default works well for standard multi-column texts.
    custom_layout_settings = LAParams(
        line_margin=0.5,  # Max distance between two lines to group into a paragraph
        word_margin=0.1,  # Max distance between characters to consider them a single word
    )

    try:
        # High-level extractor processes the entire file stream natively
        return extract_text(pdf_path, laparams=custom_layout_settings)
    except Exception:
        return ""


def extract_text_pymupdf(path: Path) -> str:
    """Extracts text from a digital PDF using PyMuPDF."""
    # Open the document
    doc = pymupdf.open(path)
    full_text = []

    for page in doc:
        # 'sort=True' forces PyMuPDF to respect natural reading layout
        # (e.g., top-to-bottom, left-to-right for multi-column research papers)
        text = page.get_text("text", sort=True)
        full_text.append(f"--- Page {page.number + 1} ---\n{text}")

    doc.close()
    return "\n".join(full_text)
