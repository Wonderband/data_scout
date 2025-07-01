import os
import glob
import gradio as gr
from PIL import Image
from dotenv import load_dotenv
from ocr.batch_ocr import process_image, ocr_and_openai, process_all_images
from db.chroma_utils import create_db, search_hybrid

load_dotenv()

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
example_images = glob.glob(os.path.join(CURR_DIR, 'demo_images', '*'))
transcript_paths = glob.glob(os.path.join(CURR_DIR, 'transcripts', '*'))

with gr.Blocks(title="Image-to-Text or OCR Demo") as scout_app:
    gr.Markdown("# Image-to-Text or OCR Demo")
    gr.Markdown("Upload an image to get a description (CLIP) or extract text (OCR).")
    transcripts_state = gr.State(value=transcript_paths)
    base_dir_state = gr.State(value=CURR_DIR)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="filepath",
                label="Upload an Image"
            )
            mode = gr.Radio(
                ["CLIP Description",
                 "EasyOCR Text Extraction",
                 "Tesseract Text Extraction",
                 "PaddleOCR Text Extraction"],
                value="CLIP Description",
                label="Choose Mode"
            )

            submit_btn = gr.Button("Analyze Image")
            compare_btn = gr.Button("Compare & Normalize with OpenAI")
            process_all = gr.Button("Process All Images")
            create_vector_db = gr.Button("Create Chroma DB")
            gr.Examples(
                examples=example_images,
                inputs=image_input,
            )

        with gr.Column():
            description_output = gr.Textbox(label="Result")
        
        # New search column
        with gr.Column():
            search_query = gr.Textbox(label="Search Query")
            search_btn = gr.Button("Search Chroma DB")
            search_output = gr.Textbox(label="Search Results")

    submit_btn.click(
        fn=process_image,
        inputs=[image_input, mode],
        outputs=[description_output]
    )
    compare_btn.click(
        fn=ocr_and_openai,
        inputs=[image_input],
        outputs=[description_output]
    )
    process_all.click(
        fn=process_all_images,
        inputs=[],
        outputs=[description_output]
    )

    create_vector_db.click(
        fn=create_db,
        inputs=[transcripts_state, base_dir_state],
        outputs=[description_output]
    )
    
    search_btn.click(
        fn=search_hybrid,
        inputs=[search_query, base_dir_state],
        outputs=[search_output]
    )

if __name__ == "__main__":
    print("Starting Gradio Blocks interface…")
    scout_app.launch(share=True)