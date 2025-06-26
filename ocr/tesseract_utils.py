import pytesseract
from pytesseract import Output
from PIL import Image


def extract_text_with_tesseract(image: Image.Image):
    """
    Extracts text from a PIL image using Tesseract OCR.
    Args:
        image (PIL.Image): The input image.
    Returns:
        str: The extracted text or a message if no text is found.
    """
    # Specify languages: 'eng+ukr+rus' for English, Ukrainian, Russian
    text = pytesseract.image_to_string(image, lang='eng+ukr+rus')
    return text.strip() if text.strip() else "No text found."


def extract_tsv_with_tesseract(image: Image.Image, langs: str = "ukr+rus") -> str:
    """
    Runs Tesseract in "tsv" mode on the given PIL image and returns the raw TSV text.

    Args:
        image (PIL.Image): The input image.
        langs (str): Languages to load (e.g., "eng+ukr+rus").

    Returns:
        str: The full TSV output (each line is a tab-separated record).
    """

    tsv_data = pytesseract.image_to_data(image, lang=langs, output_type=Output.STRING)
    return tsv_data
