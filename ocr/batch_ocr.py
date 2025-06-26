import os
import json
from PIL import Image

from openai import OpenAI
from ocr.easyocr_utils import extract_text_with_easyocr
from ocr.tesseract_utils import extract_text_with_tesseract, extract_tsv_with_tesseract
from clip.clip_utils import process_image_with_clip


openai_api_key = os.getenv("OPENAI_API_KEY")
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
TRANS_DIR = os.path.join(CURR_DIR, "..", "transcripts")
DEMO_DIR = os.path.join(CURR_DIR, "..", "demo_images")
os.makedirs(TRANS_DIR, exist_ok=True)


def process_image(image_path: str, mode: str) -> str:
    img = Image.open(image_path)
    if mode == "CLIP Description":
        return process_image_with_clip(img)
    elif mode == "EasyOCR Text Extraction":
        return extract_text_with_easyocr(img)
    elif mode == "Tesseract Text Extraction":
        return extract_text_with_tesseract(img)
    else:
        return "Unknown mode selected."


def ocr_and_openai(image_path: str) -> json:
    image_filename = os.path.basename(image_path)
    name_root, _ext = os.path.splitext(image_filename)
    img = Image.open(image_path)
    easy = extract_text_with_easyocr(img)
    tess = extract_text_with_tesseract(img)

    client = OpenAI(api_key=openai_api_key)

    prompt = (
        "You are an expert in document data extraction and normalization. "
        "Given two OCR outputs of the same invoice, your task is to:\n"
        "- Fix all mistakes in the texts.\n"
        "- Extract all relevant fields and their values, preserving the original field names.\n"
        "- Never add or remove any information. Use only the data you received from 2 OCR outputs! No other words are allowed!\n"
        "- Normalize all numbers (remove spaces, use dot as decimal separator if needed) and dates (use YYYY-MM-DD format).\n"
        "- If a field is present in both, use the most confident or complete value, or merge if needed.\n"
        "- Return a single resulting JSON object with all extracted fields and values.\n\n"
        "OCR result 1 (EasyOCR):\n"
        f"{easy}\n\n"
        "OCR result 2 (Tesseract):\n"
        f"{tess}\n\n"
        "Return only the resulting JSON object, without any ''' at the start and ending and without `json` word itself "
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant for invoice data extraction and normalization."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.1,
        )
        result_json = response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"

    json_path = os.path.join(TRANS_DIR, f"{name_root}.json")
    try:
        parsed = json.loads(result_json)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result_json)

    return result_json


def process_all_images() -> str:
    """
    Loops through all images in demo_images and runs ocr_and_openai on each.
    """

    example_images = [
        os.path.join(DEMO_DIR, f)
        for f in os.listdir(DEMO_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]
    results = []
    for img_path in example_images:
        res = ocr_and_openai(img_path)
        results.append(res)
    return "\n".join(results)
