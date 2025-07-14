import os
import glob
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from ocr.batch_ocr import process_image, ocr_and_openai, process_all_images
from db.chroma_utils import create_db
from rag.ai_processor import perform_rag, process_table

load_dotenv()

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
example_images = glob.glob(os.path.join(CURR_DIR, 'demo_images', '*'))
transcript_paths = glob.glob(os.path.join(CURR_DIR, 'transcripts', 'acts','*'))


def load_excel(file_obj):
    """Read the uploaded .xlsx/.csv into a pandas DataFrame."""
    df = pd.read_excel(file_obj.name)  # requires openpyxl for .xlsx
    # Return: 1) the DataFrame, 2) make it visible
    return df, gr.update(visible=True)


def save_changes(df_edited):
    """Write the edited DataFrame back to disk."""
    out_path = os.path.join(CURR_DIR, "edited_data.xlsx")
    df_edited.to_excel(out_path, index=False)
    return f"Saved changes to {out_path}"


with gr.Blocks(title="Image‑to‑Text or OCR Demo") as scout_app:
    gr.Markdown("# Document AUDITOR")
    gr.Markdown("Upload an image to get a description (CLIP) or extract text (OCR).")
    transcripts_state = gr.State(value=transcript_paths)
    base_dir_state = gr.State(value=CURR_DIR)

    # ─── IMAGE / OCR / RAG ROW ────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload an Image")
            mode = gr.Radio(
                ["CLIP Description",
                 "EasyOCR Text Extraction",
                 "Tesseract Text Extraction",
                 # "PaddleOCR Text Extraction"
                 ],
                value="CLIP Description",
                label="Choose Mode",
            )
            # submit_btn    = gr.Button("Analyze Image")
            compare_btn = gr.Button("Compare & Normalize with OpenAI")
            process_all = gr.Button("Process All Images")
            create_vector = gr.Button("Create Chroma DB")
            gr.Examples(examples=example_images, inputs=image_input)

        with gr.Column():
            description_output = gr.Textbox(label="Result")

        with gr.Column():
            search_query = gr.Textbox(label="Search Query")
            search_btn = gr.Button("Search Chroma DB")
            search_output = gr.Textbox(label="Search Results")

    # submit_btn.click(process_image,    [image_input, mode],           [description_output])
    compare_btn.click(ocr_and_openai, [image_input], [description_output])
    process_all.click(process_all_images, [], [description_output])
    create_vector.click(create_db, [transcripts_state, base_dir_state], [description_output])
    search_btn.click(perform_rag, [search_query, base_dir_state], [search_output])

    # ─── EXCEL UPLOAD & EDITABLE TABLE ───────────────────────────────
    gr.Markdown("## Editable Table from Excel")
    with gr.Row():
        excel_uploader = gr.UploadButton(
            label="Upload Excel or CSV",
            file_types=['.xlsx', '.csv'],
            file_count="single",
        )

    with gr.Row():
        excel_df = gr.Dataframe(
            datatype="str",
            interactive=True,
            visible=False,  # hidden until upload
            label="Click into any cell to edit",
        )

    # When a file is uploaded → load it into excel_df & show the grid
    excel_uploader.upload(
        fn=load_excel,
        inputs=[excel_uploader],
        outputs=[excel_df, excel_df]
    )

    # ─── PROCESS & SAVE ROW ───────────────────────────────────────────
    with gr.Row():
        process_table_btn = gr.Button("Process Entire Table")
        save_btn = gr.Button("Save changes back to disk")
        save_status = gr.Textbox(label="Save status")

    # Process the currently visible & edited table
    process_table_btn.click(
        fn=process_table,
        inputs=[excel_df, base_dir_state],
        outputs=[excel_df]
    )

    # Save whatever is in excel_df right now
    save_btn.click(
        fn=save_changes,
        inputs=[excel_df],
        outputs=[save_status]
    )

if __name__ == "__main__":
    print("Starting Gradio Blocks interface…")
    scout_app.launch(share=True, debug=True)
