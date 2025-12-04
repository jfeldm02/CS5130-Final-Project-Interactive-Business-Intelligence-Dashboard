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
            gr.Markdown("""
                ## Filter and Transform Your Data
                Build transformations interactively. Changes preview live as you configure them.
                **Disclaimer:** Perform the change name operation last if desired! The other tabs cannot track the new name column.
            """)
            
            # State for tracking operations
            pending_operations = gr.State([])
            
            with gr.Row():
                # Left panel: Controls
                with gr.Column(scale=1):
                    filter_file_dropdown = gr.Dropdown(
                        label="Choose Dataset",
                        choices=[],
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### Add Operation")
                    
                    operation_type = gr.Radio(
                        label="Operation Type",
                        choices=["Sort", "Filter (Range)", "Filter (Values)", "Rename Column", "Select Columns"],
                        value="Sort"
                    )
                    
                    # Sort inputs
                    with gr.Group() as sort_group:
                        sort_columns = gr.Dropdown(
                            label="Sort by Column(s)",
                            choices=[],
                            multiselect=True,
                            allow_custom_value=False,
                            interactive=True
                        )
                        sort_order = gr.Radio(
                            label="Order",
                            choices=["Ascending", "Descending"],
                            value="Ascending"
                        )
                    
                    # Range filter inputs
                    with gr.Group(visible=False) as range_group:
                        range_column = gr.Dropdown(
                            label="Column",
                            choices=[],
                            allow_custom_value=False,
                            interactive=True
                        )
                        range_min = gr.Number(label="Min", value=None)
                        range_max = gr.Number(label="Max", value=None)
                    
                    # Value filter inputs
                    with gr.Group(visible=False) as values_group:
                        values_column = gr.Dropdown(
                            label="Column",
                            choices=[],
                            allow_custom_value=False,
                            interactive=True
                        )
                        available_values = gr.Dropdown(
                            label="Keep Values",
                            choices=[],
                            multiselect=True,
                            allow_custom_value=False,
                            interactive=True
                        )
                    
                    # Rename inputs
                    with gr.Group(visible=False) as rename_group:
                        rename_old = gr.Dropdown(
                            label="Column to Rename",
                            choices=[],
                            allow_custom_value=False,
                            interactive=True
                        )
                        rename_new = gr.Textbox(
                            label="New Name",
                            placeholder="Enter new name..."
                        )
                    
                    # Select columns inputs
                    with gr.Group(visible=False) as select_group:
                        select_columns = gr.Dropdown(
                            label="Columns to Keep",
                            choices=[],
                            multiselect=True,
                            allow_custom_value=False,
                            interactive=True
                        )
                    
                    add_operation_btn = gr.Button("+ Add Operation", variant="secondary")
                    
                    gr.Markdown("---")
                    
                    operations_summary = gr.Textbox(
                        label="Pending Operations",
                        value="No operations added yet",
                        interactive=False,
                        lines=8
                    )
                    
                    with gr.Row():
                        clear_operations_btn = gr.Button("Clear All", size="sm")
                        undo_operation_btn = gr.Button("Undo Last", size="sm")
                
                # Right panel: Live preview
                with gr.Column(scale=2):
                    gr.Markdown("### Live Preview")
                    
                    preview_stats = gr.Markdown("*Select a dataset to begin*")
                    
                    operations_preview = gr.DataFrame(
                        label="Data Preview (first 20 rows)",
                        interactive=False,
                        wrap=True
                    )
                    
                    gr.Markdown("---")
                    
                    with gr.Row():
                        save_config_name = gr.Textbox(
                            label="Save As (optional)",
                            placeholder="e.g., sales_filtered",
                            scale=2
                        )
                        apply_operations_btn = gr.Button(
                            "Save Transformed Dataset",
                            variant="primary",
                            scale=1
                        )
        
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
        
        # ==========================================
        # Filter & Explore handlers
        # ==========================================
        
        # Toggle visibility of operation input groups based on selected operation type
        def toggle_operation_groups(op_type):
            return (
                gr.Group(visible=(op_type == "Sort")),
                gr.Group(visible=(op_type == "Filter (Range)")),
                gr.Group(visible=(op_type == "Filter (Values)")),
                gr.Group(visible=(op_type == "Rename Column")),
                gr.Group(visible=(op_type == "Select Columns"))
            )
        
        operation_type.change(
            fn=toggle_operation_groups,
            inputs=[operation_type],
            outputs=[sort_group, range_group, values_group, rename_group, select_group]
        )
        
        # Load columns when dataset is selected
        filter_file_dropdown.change(
            fn=utils.load_filter_columns_and_preview,
            inputs=[loaded_data, filter_file_dropdown, pending_operations],
            outputs=[
                sort_columns, range_column, values_column, rename_old, select_columns,
                operations_preview, preview_stats, pending_operations, operations_summary
            ]
        )
        
        # Load unique values when values_column changes
        values_column.change(
            fn=utils.load_unique_values,
            inputs=[loaded_data, filter_file_dropdown, values_column],
            outputs=[available_values]
        )
        
        # Add operation to pending list
        add_operation_btn.click(
            fn=utils.add_operation,
            inputs=[
                pending_operations, operation_type,
                sort_columns, sort_order,
                range_column, range_min, range_max,
                values_column, available_values,
                rename_old, rename_new,
                select_columns,
                loaded_data, filter_file_dropdown
            ],
            outputs=[pending_operations, operations_summary, operations_preview, preview_stats]
        )
        
        # Clear all operations
        clear_operations_btn.click(
            fn=utils.clear_operations,
            inputs=[loaded_data, filter_file_dropdown],
            outputs=[pending_operations, operations_summary, operations_preview, preview_stats]
        )
        
        # Undo last operation
        undo_operation_btn.click(
            fn=utils.undo_operation,
            inputs=[pending_operations, loaded_data, filter_file_dropdown],
            outputs=[pending_operations, operations_summary, operations_preview, preview_stats]
        )
        
        # Apply and save operations
        apply_operations_btn.click(
            fn=utils.save_transformed_dataset,
            inputs=[loaded_data, filter_file_dropdown, pending_operations, save_config_name],
            outputs=[
                loaded_data, pending_operations, operations_summary, 
                operations_preview, preview_stats, filter_file_dropdown
            ]
        )

    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch()