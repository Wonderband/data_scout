#!/usr/bin/env python3
import glob
import os
from pdf2image import convert_from_path

input_dir = os.path.join(os.getcwd(), "..", "demo_images")

for pdf_path in glob.glob(os.path.join(input_dir, "*.pdf")):
    # convert each page to a list of PIL images
    pages = convert_from_path(pdf_path, dpi=200)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # save each page as JPEG in the same folder
    for i, page in enumerate(pages, start=1):
        jpg_path = os.path.join(input_dir, f"{base_name}_page{i}.jpg")
        page.save(jpg_path, "JPEG")
        print(f"Saved {jpg_path}")
