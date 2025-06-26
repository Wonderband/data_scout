import easyocr
from PIL import Image
import numpy as np

# Initialize the EasyOCR reader (English, Ukrainian, and Russian languages)
reader = easyocr.Reader(['en', 'uk', 'ru'], gpu=False)


def extract_text_with_easyocr(image: Image.Image) -> str:
    """
    Extracts text from a PIL image using EasyOCR.
    Args:
        image (PIL.Image): The input image.
    Returns:
        str: The extracted text or a message if no text is found.
    """
    # Convert PIL image to numpy array
    img_np = np.array(image)
    results = reader.readtext(img_np)
    if not results:
        return "No text found."
    # Join all detected text segments
    extracted = " ".join([text for _, text, _ in results])
    return extracted


