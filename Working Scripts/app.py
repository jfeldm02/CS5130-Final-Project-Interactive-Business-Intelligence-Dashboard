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
            gr.Markdown("""
                ## Step 1: Load Your Dataset(s)
                ### Choose a folder or a single file containing data you would like to analyze. You will use this loaded data throughout the other tabs!
                Note: Only the following file types are supported: .csv, .tsv, .xlsx, 
                .xls, .json, .h5, .hdf5, .parquet, .feather
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    input_directory = gr.File(
                        label="Upload Your File or Folder",
                        file_count="multiple",
                        height=400
                    )
                    
                    load_btn = gr.Button("Load and Import Data", variant="primary", size="lg")

                    load_status = gr.Textbox(
                        label="Load Status", 
                        interactive=False, 
                        lines=10
                    )

            gr.Markdown("""
                ## Step 2: Preview Your Dataset(s)
                ### Choose a loaded file from the dropdown and preview the head and tail. 
                Note: This is a good time to see if your raw data needs to be better formatted before
                uploading.     
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    preview_file_dropdown = gr.Dropdown(
                        label="Choose Loaded Dataset",
                        choices=[],  # Update dynamically
                        allow_custom_value=False,
                        interactive=True
                    )

                    preview_btn = gr.Button(
                        "Generate Preview", 
                        variant="secondary"
                        )
                    
                    gr.Markdown("""
                                ### Head/Tail Preview
                                """)
                    
                    preview_output = gr.DataFrame(
                        interactive=False,
                        wrap=True
                    )
            
            gr.Markdown("""
                ## Continue to the Statistics Tab!   
            """)

        with gr.Tab("Statistics"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                                ## Step 3: Profile the Dataset
                                ### Select a loaded file from the dropdown to generate a profile summary.
                                """)

                    # Dropdown populated dynamically with loaded file names
                    profile_file_dropdown = gr.Dropdown(
                        label="Choose Loaded Dataset",
                        choices=[],  # Update dynamically
                        allow_custom_value=False,
                        interactive=True
                    )

                    profile_btn = gr.Button(
                        "Generate Profile", 
                        variant="secondary"
                        )

                    gr.Markdown("""
                                ### Dataset Summary
                                """)
                    
                    profile_output_summary = gr.DataFrame(
                        interactive=False,
                        wrap=True
                    )

                    gr.Markdown("""
                                ### Column Summary
                                """)
                    
                    profile_output_columns = gr.DataFrame(
                        interactive=False,
                        wrap=True
                    )
            gr.Markdown("""
            ## Continue to the Clean Data Tab!   
            """)

        with gr.Tab("Clean Data"):
            # Interactive filtering
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                                ## Step 4: Clean The Dataset
                                ### This is highly recommended before performing any further analysis.
                                Disclaimer: Edits made here edit the original file in place. 
                                """)
                    
                    gr.Markdown("""
                                ### a. Change column types:
                                View the current column datatypes below and change them to the desired consistent datatype. 
                                Disclaimer: This will turn any cell that is unable to be converted into a Null. (i.e. Strings cannot be converted to floats)
                                """)

                    # Dropdown to select dataset
                    dtype_file_dropdown = gr.Dropdown(
                        label="Choose Dataset",
                        choices=[],
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    view_dtypes_btn = gr.Button("View Data Types", variant="secondary")
                    
                    # Display current data types
                    dtype_display = gr.DataFrame(
                        label="Current Column Data Types",
                        interactive=False,
                        wrap=True
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Convert Column")
                    
                    # Multi-select for columns
                    columns_to_convert = gr.Dropdown(
                        label="Select Column(s) to Convert",
                        choices=[],
                        multiselect=True,
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    # Dropdown for target data type
                    target_dtype = gr.Dropdown(
                        label="Target Data Type",
                        choices=[
                            "int32", "int64", 
                            "float32", "float64",
                            "str", "object",
                            "category",
                            "datetime64[ns]",
                            "bool"
                        ],
                        value="float64",
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    convert_btn = gr.Button("Convert Data Type", variant="primary")
                    
                    conversion_status = gr.Textbox(
                        label="Conversion Status",
                        interactive=False,
                        lines=3
                    )

            gr.Markdown("---")  # Separator
    
            gr.Markdown("""
                ## b. Fill Null Values
                ### Select columns with null values and choose a fill method.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Dropdown to select dataset
                    null_file_dropdown = gr.Dropdown(
                        label="Choose Dataset",
                        choices=[],
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    view_nulls_btn = gr.Button("View Columns with Nulls", variant="secondary")
                    
                    # Display null counts
                    null_info_display = gr.DataFrame(
                        label="Null Value Summary",
                        interactive=False,
                        wrap=True
                    )
                    
                    # Multi-select for columns with nulls
                    null_columns_dropdown = gr.Dropdown(
                        label="Select Column(s) to Fill",
                        choices=[],
                        multiselect=True,
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    # Fill method dropdown
                    fill_method_dropdown = gr.Dropdown(
                        label="Fill Method",
                        choices=[
                            "mean",
                            "median", 
                            "mode",
                            "random"
                        ],
                        value="mean",
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    gr.Markdown("""
                    **Fill Methods:**
                    - **mean**: Fill with column average (numeric only)
                    - **median**: Fill with column median (numeric only)
                    - **mode**: Fill with most common value
                    - **random**: Fill with random value from column range
                    """)
                    
                    fill_nulls_btn = gr.Button("Fill Null Values", variant="primary")
                    
                    null_status = gr.Textbox(
                        label="Fill Status",
                        interactive=False,
                        lines=8
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Updated Dataset Summary")
                    
                    null_summary = gr.DataFrame(
                        label="Overall Summary",
                        interactive=False,
                        wrap=True
                    )
                    
                    null_col_stats = gr.DataFrame(
                        label="Column Statistics",
                        interactive=False,
                        wrap=True
                    )

                    gr.Markdown("---")

                    gr.Markdown("""
                                    ### c. Drop duplicate rows if desired:
                                    View the number duplicate rows in your dataset and remove them. 
                                    """)
                
            with gr.Row():
                with gr.Column(scale=1):
                    # Dropdown to select dataset
                    duplicates_dropdown = gr.Dropdown(
                        label="Choose Dataset",
                        choices=[],
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    drop_dupes_btn = gr.Button("Drop Duplicates", variant="primary")
                    
                    duplicates_status = gr.Textbox(
                        label="Cleaning Status",
                        interactive=False,
                        lines=5
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Updated Dataset Summary")
                    
                    cleaned_summary = gr.DataFrame(
                        label="Overall Summary",
                        interactive=False,
                        wrap=True
                    )
                    
                    cleaned_col_stats = gr.DataFrame(
                        label="Column Statistics",
                        interactive=False,
                        wrap=True
                    )
    
            # Update dtype_file_dropdown when data is loaded
            load_btn.click(
                fn=utils.data_upload_pipeline,
                inputs=[input_directory],
                outputs=[load_status,
                        loaded_data,
                        preview_file_dropdown,
                        profile_file_dropdown,
                        dtype_file_dropdown,
                        duplicates_dropdown,
                        null_file_dropdown] 
            )

            view_dtypes_btn.click(
                fn=utils.update_dtype_view_and_columns,
                inputs=[loaded_data, dtype_file_dropdown],
                outputs=[dtype_display, columns_to_convert]
            )
            
            # Convert button
            convert_btn.click(
                fn=utils.convert_dtype_wrapper,
                inputs=[loaded_data, dtype_file_dropdown, columns_to_convert, target_dtype],
                outputs=[conversion_status, dtype_display, loaded_data]
            )

            # View columns with nulls
            def update_null_columns(loaded_data, selected_file):
                columns = utils.get_columns_with_nulls(loaded_data, selected_file)
                return gr.Dropdown(choices=columns)
            
            view_nulls_btn.click(
                fn=update_null_columns,
                inputs=[loaded_data, null_file_dropdown],
                outputs=[null_info_display, null_columns_dropdown]
            )
            
            # Fill nulls button
            fill_nulls_btn.click(
                fn=utils.fill_nulls_wrapper,
                inputs=[loaded_data, null_file_dropdown, null_columns_dropdown, fill_method_dropdown],
                outputs=[null_status, null_summary, null_col_stats, loaded_data]
            )
            drop_dupes_btn.click(
                fn=utils.drop_duplicates_wrapper,
                inputs=[loaded_data, duplicates_dropdown],
                outputs=[duplicates_status, cleaned_summary, cleaned_col_stats, loaded_data]
            )

        with gr.Tab("Filter & Explore"):
            # Interactive filtering
            pass
                    

    
        
        with gr.Tab("Visualizations"):
            # Charts and graphs
            pass
        
        with gr.Tab("Insights"):
            # Automated insights
            pass
            
            load_btn.click(
                fn=utils.data_upload_pipeline,
                inputs=[input_directory],
                outputs=[load_status,
                         loaded_data,
                         preview_file_dropdown,
                         profile_file_dropdown]
            )

            preview_btn.click(
                fn=dp.preview_file,
                inputs=[loaded_data, preview_file_dropdown],
                outputs=[preview_output]
            )

            profile_btn.click(
                fn=utils.profile_file,
                inputs=[loaded_data, profile_file_dropdown],
                outputs=[profile_output_summary, profile_output_columns]
            )

    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch()