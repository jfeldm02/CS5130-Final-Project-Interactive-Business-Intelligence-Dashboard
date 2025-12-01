import gradio as gr
import pandas as pd

import data_processor as dp
import utils

def create_dashboard():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
            # Interactive Business Intelligence Dashboard
            ### Justin Feldman | CS5130: Final Project | Professor Lino Coria Mendoza
            """)
        
        # I was unfamiliar with gr.State so this was recommended by AI
        loaded_data = gr.State({})

        with gr.Tab("Data Upload"):
            # Step 1: Dataset loading section
            gr.Markdown("""
            ## Step 1: Load the Dataset
            ### Choose a folder or a single file containing data you would like to analyze.
            Note: Only the following file types are supported: .csv, .tsv, .xlsx, 
            .xls, .json, .h5, .hdf5, .parquet, .feather
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    input_directory = gr.File(
                        label="Upload Your File or Folder",
                        file_count="directory",
                        height=400
                        )
                    
                    load_btn = gr.Button("Load and Import Data", variant="primary", size="lg")

                    load_status = gr.Textbox(
                        label="Load Status", 
                        interactive=False, 
                        lines=10
                        )

            load_btn.click(
                fn=utils.data_upload_pipeline,
                inputs=[input_directory],
                outputs=[load_status, loaded_data]
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                                ## Step 2: Profile the Dataset
                                ### Select a loaded file from the dropdown to generate a profile summary.
                                """)

                    # Dropdown populated dynamically with loaded file names
                    profile_file_dropdown = gr.Dropdown(
                        label="Choose Loaded Dataset",
                        choices=[],  # Update dynamically
                        allow_custom_value=False,
                        interactive=True
                    )

                    loaded_data.change(
                        fn=utils.update_dropdown_choices,
                        inputs=[loaded_data],
                        outputs=[profile_file_dropdown]
                    )

                    profile_btn = gr.Button("Generate Profile", variant="secondary")
                    profile_output = gr.Markdown(label="Profile Summary", elem_id="profile_output")

                    profile_btn.click(
                        fn=utils.profile_file,
                        inputs=[loaded_data, profile_file_dropdown],
                        outputs=[profile_output]
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