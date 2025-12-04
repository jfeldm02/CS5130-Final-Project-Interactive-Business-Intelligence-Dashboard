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

        with gr.Tab("Statistics & Data Cleaning"):
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
            ## Let's clean the data and check these stats again!   
            """)
            gr.Markdown("---")  
            gr.Markdown("---")  

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

            gr.Markdown("---")  
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
                    - **remove**: Drops the row where nulls exist in the column
                    """)
                    
                    fill_nulls_btn = gr.Button("Fill Null Values", variant="primary")
                    
                    null_status = gr.Textbox(
                        label="Fill Status",
                        interactive=False,
                        lines=8
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

            gr.Markdown("---")
            gr.Markdown("---")
            gr.Markdown("## Updated Dataset Summary")

            with gr.Row():
                with gr.Column(scale=1):
                    
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

            gr.Markdown("---")

            gr.Markdown("""
                ## Download Cleaned Dataset
                ### Export your cleaned dataset with the suffix '_cleaned' added to the filename.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    # Dropdown to select dataset to download
                    download_file_dropdown = gr.Dropdown(
                        label="Choose Dataset to Download",
                        choices=[],
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    download_btn = gr.Button("Download Cleaned Dataset", variant="primary")
                    
                    download_output = gr.File(
                        label="Download File",
                        interactive=False
                    )
            
        with gr.Tab("Filter & Explore"):
            # Interactive filtering
            # Update the Filter & Transform tab
            gr.Markdown("""
                ## Filter and Transform Your Data
                ### Apply multiple operations: sort, filter by range, filter by values, rename columns, and select columns.
                **All operations create a new dataset, preserving the original.**
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Dataset selection
                    filter_file_dropdown = gr.Dropdown(
                        label="Choose Dataset",
                        choices=[],
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    load_columns_btn = gr.Button("Load Columns", variant="secondary")
            
            # Sort Section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Sort Data")
                    
                    sort_columns = gr.Dropdown(
                        label="Sort by Column(s)",
                        choices=[],
                        multiselect=True,
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    sort_order = gr.Radio(
                        label="Sort Order",
                        choices=["Ascending", "Descending"],
                        value="Ascending"
                    )
            
            # Filter by Range Section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Filter by Numeric Range")
                    
                    range_column = gr.Dropdown(
                        label="Column to Filter",
                        choices=[],
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    with gr.Row():
                        range_min = gr.Number(
                            label="Minimum Value (leave empty for no minimum)",
                            value=None
                        )
                        range_max = gr.Number(
                            label="Maximum Value (leave empty for no maximum)",
                            value=None
                        )
            
            # Filter by Values Section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 3. Filter by Specific Values")
                    
                    values_column = gr.Dropdown(
                        label="Column to Filter",
                        choices=[],
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    available_values = gr.Dropdown(
                        label="Select Values to Keep",
                        choices=[],
                        multiselect=True,
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    load_values_btn = gr.Button("Load Available Values", size="sm")
            
            # Rename Columns Section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 4. Rename Columns")
                    gr.Markdown("*Add multiple renames by filling fields and clicking 'Add Rename'. View all queued renames below.*")
                    
                    with gr.Row():
                        rename_old = gr.Dropdown(
                            label="Column to Rename",
                            choices=[],
                            allow_custom_value=False,
                            interactive=True
                        )
                        
                        rename_new = gr.Textbox(
                            label="New Name",
                            placeholder="Enter new column name..."
                        )
                    
                    add_rename_btn = gr.Button("Add Rename to Queue", size="sm", variant="secondary")
                    
                    rename_queue = gr.State({})  # Store rename mappings
                    
                    rename_queue_display = gr.Textbox(
                        label="Queued Column Renames",
                        value="No renames queued",
                        interactive=False,
                        lines=4
                    )
                    
                    clear_renames_btn = gr.Button("Clear Rename Queue", size="sm")
            
            # Select Columns Section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 5. Select Columns to Keep")
                    
                    select_columns = gr.Dropdown(
                        label="Columns to Keep (leave empty to keep all)",
                        choices=[],
                        multiselect=True,
                        allow_custom_value=False,
                        interactive=True
                    )
            
            # Preview Results and Save
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Result Preview (first 20 rows)")
                    
                    operations_preview = gr.DataFrame(
                        label="Transformed Data Preview",
                        interactive=False,
                        wrap=True
                    )
            
            # Save Configuration Section
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                        ### 6. Name Your Transformed Dataset (Optional)
                        **Leave empty** to auto-generate a name like `dataset_transformed`, or 
                        **provide a custom name** to easily reference this configuration later.
                    """)
                    
                    save_config_name = gr.Textbox(
                        label="Custom Dataset Name (optional)",
                        placeholder="e.g., sales_q4_cleaned, high_value_customers, etc.",
                        interactive=True
                    )
            
            # Apply Operations
            with gr.Row():
                with gr.Column(scale=1):
                    apply_operations_btn = gr.Button(
                        "Apply Operations & Save", 
                        variant="primary",
                        size="lg"
                    )
                    
                    operations_status = gr.Textbox(
                        label="Operations Status",
                        interactive=False,
                        lines=10
                    )
            
            # Event handlers need to go AFTER all tabs are defined
            # (Move these to after all your tabs are created, before demo.launch())
        
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
                    profile_file_dropdown,
                    dtype_file_dropdown,
                    duplicates_dropdown,
                    null_file_dropdown,
                    download_file_dropdown,
                    filter_file_dropdown]
        )

        preview_btn.click(
            fn=dp.preview_file,
            inputs=[loaded_data, preview_file_dropdown],
            outputs=[preview_output]
        )

        # Statistics handlers
        profile_btn.click(
            fn=utils.profile_file,
            inputs=[loaded_data, profile_file_dropdown],
            outputs=[profile_output_summary, profile_output_columns]
        )
        
        view_dtypes_btn.click(
            fn=utils.update_dtype_view_and_columns,
            inputs=[loaded_data, dtype_file_dropdown],
            outputs=[dtype_display, columns_to_convert]
        )

        convert_btn.click(
            fn=utils.convert_dtype_wrapper,
            inputs=[loaded_data, dtype_file_dropdown, columns_to_convert, target_dtype],
            outputs=[conversion_status, dtype_display, loaded_data]
        )
        
        view_nulls_btn.click(
            fn=utils.get_columns_with_nulls,
            inputs=[loaded_data, null_file_dropdown],
            outputs=[null_info_display, null_columns_dropdown]
        )
        
        fill_nulls_btn.click(
            fn=utils.fill_nulls_wrapper,
            inputs=[loaded_data, null_file_dropdown, null_columns_dropdown, fill_method_dropdown],
            outputs=[null_status, loaded_data] 
        )

        drop_dupes_btn.click(
            fn=utils.drop_duplicates_wrapper,
            inputs=[loaded_data, duplicates_dropdown],
            outputs=[duplicates_status, cleaned_summary, cleaned_col_stats, loaded_data]
        )
        
        download_btn.click(
            fn=utils.prepare_download,
            inputs=[loaded_data, download_file_dropdown],
            outputs=[download_output]
        )
        
        # Filter & Explore handlers
        load_columns_btn.click(
            fn=utils.update_filter_columns,
            inputs=[loaded_data, filter_file_dropdown],
            outputs=[sort_columns, range_column, values_column, select_columns, rename_old]
        )

        load_values_btn.click(
            fn=utils.load_unique_values,
            inputs=[loaded_data, filter_file_dropdown, values_column],
            outputs=[available_values]
        )

        add_rename_btn.click(
            fn=utils.add_to_rename_queue,
            inputs=[rename_queue, rename_old, rename_new],
            outputs=[rename_queue, rename_queue_display]
        )

        clear_renames_btn.click(
            fn=utils.clear_rename_queue,
            inputs=[],
            outputs=[rename_queue, rename_queue_display]
        )

        apply_operations_btn.click(
            fn=utils.apply_all_operations,  # Fixed function name
            inputs=[loaded_data, filter_file_dropdown, sort_columns, sort_order,
                    range_column, range_min, range_max, values_column, available_values,
                    rename_queue, select_columns, save_config_name], 
            outputs=[operations_status, operations_preview, loaded_data]
        )

    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch()