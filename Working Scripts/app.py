import gradio as gr
import pandas as pd
from data_processor import *

def create_dashboard():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Business Intelligence Dashboard")
        
        with gr.Tab("Data Upload"):
            # File upload and preview
            gr.Markdown("""
            # Interactive Business Intelligence Dashboard
            ### Justin Feldman | CS5130: Final Project | Professor Lino Coria Mendoza
            """)

            # Step 1: Dataset loading section
            gr.Markdown("""
            ## Step 1: Load the Dataset
            ### Choose a folder or a single file containing data you would like to analyze.
            Note: Only the following file types are supported: .csv, .tsv, .xlsx, 
            .xls, .json, .h5, .hdf5, .parquet, .feather
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    input_directory = gr.File(label="Upload Your File or Folder",
                                              file_count="directory",
                                              height=400)
                    load_btn = gr.Button("Load and Import Data", variant="primary", size="lg")
                    load_status = gr.Textbox(label="Load Status", interactive=False)

            load_btn.click(
                fn=summarize_directory,
                inputs=[],
                outputs=[dataset_status]
            )

        with gr.Tab("Statistics"):
            # Summary statistics and profiling
            pass
        
        with gr.Tab("Filter & Explore"):
            # Interactive filtering
            pass
        
        with gr.Tab("Visualizations"):
            # Charts and graphs
            pass
        
        with gr.Tab("Insights"):
            # Automated insights
            pass
    
    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch()