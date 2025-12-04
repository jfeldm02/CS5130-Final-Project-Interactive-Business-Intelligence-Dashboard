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
                ## Load Your Dataset(s)
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
                ## Preview Your Dataset(s)
                ### Choose a loaded file from the dropdown and preview the head and tail. 
                Note: Check if your raw data is formatted appropriately.    
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

        with gr.Tab("Statistics & Data Cleaning"):
            gr.Markdown("""
                ## Profile and Clean Your Data
                Select a dataset to view statistics and apply cleaning operations. Changes are applied in place.
            """)
            
            with gr.Row():
                # Left panel: Controls
                with gr.Column(scale=1):
                    profile_file_dropdown = gr.Dropdown(
                        label="Choose Dataset",
                        choices=[],
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### Cleaning Operations")
                    
                    cleaning_operation = gr.Radio(
                        label="Operation",
                        choices=["Convert Data Types", "Fill Null Values", "Drop Duplicates"],
                        value="Convert Data Types"
                    )
                    
                    # Convert Data Types inputs
                    with gr.Group() as dtype_group:
                        columns_to_convert = gr.Dropdown(
                            label="Column(s) to Convert",
                            choices=[],
                            multiselect=True,
                            allow_custom_value=False,
                            interactive=True
                        )
                        target_dtype = gr.Dropdown(
                            label="Target Type",
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
                        convert_btn = gr.Button("Convert", variant="primary")
                    
                    # Fill Nulls inputs
                    with gr.Group(visible=False) as nulls_group:
                        null_columns_dropdown = gr.Dropdown(
                            label="Column(s) to Fill",
                            choices=[],
                            multiselect=True,
                            allow_custom_value=False,
                            interactive=True
                        )
                        fill_method_dropdown = gr.Dropdown(
                            label="Fill Method",
                            choices=["mean", "median", "mode", "random", "remove"],
                            value="mean",
                            allow_custom_value=False,
                            interactive=True
                        )
                        gr.Markdown("""
                        *mean/median*: numeric only | *mode*: most common | *random*: from range | *remove*: drop row
                        """)
                        fill_nulls_btn = gr.Button("Fill Nulls", variant="primary")
                    
                    # Drop Duplicates inputs
                    with gr.Group(visible=False) as dupes_group:
                        gr.Markdown("Remove all duplicate rows from the dataset.")
                        drop_dupes_btn = gr.Button("Drop Duplicates", variant="primary")
                    
                    gr.Markdown("---")
                    
                    cleaning_log = gr.Textbox(
                        label="Operation Log",
                        value="Select a dataset to begin",
                        interactive=False,
                        lines=6
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### Export")
                    
                    download_btn = gr.Button("Download Cleaned Dataset", variant="secondary")
                    download_output = gr.File(label="Download", interactive=False)
                
                # Right panel: Statistics display
                with gr.Column(scale=2):
                    gr.Markdown("### Dataset Overview")
                    
                    stats_summary = gr.Markdown("*Select a dataset to view statistics*")
                    
                    profile_output_summary = gr.DataFrame(
                        label="Summary",
                        interactive=False,
                        wrap=True
                    )
                    
                    gr.Markdown("### Column Details")
                    
                    with gr.Tabs():
                        with gr.Tab("Statistics"):
                            profile_output_columns = gr.DataFrame(
                                label="Column Statistics",
                                interactive=False,
                                wrap=True
                            )
                        
                        with gr.Tab("Data Types"):
                            dtype_display = gr.DataFrame(
                                label="Column Data Types",
                                interactive=False,
                                wrap=True
                            )
                        
                        with gr.Tab("Null Values"):
                            null_info_display = gr.DataFrame(
                                label="Null Value Summary",
                                interactive=False,
                                wrap=True
                            )
                    
                    gr.Markdown("### Data Preview")
                    cleaning_preview = gr.DataFrame(
                        label="First 10 Rows",
                        interactive=False,
                        wrap=True
                    )
            
        with gr.Tab("Filter & Explore"):
            gr.Markdown("""
                ## Filter and Transform Your Data
                Build transformations interactively. Changes preview live as you configure them.
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
            gr.Markdown("""
                ## Create Visualizations
                ### Select a dataset and configure your chart settings.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Dataset selection
                    viz_file_dropdown = gr.Dropdown(
                        label="Choose Dataset",
                        choices=[],
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    load_viz_columns_btn = gr.Button("Load Columns", variant="secondary")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Chart Configuration")
                    
                    plot_type_dropdown = gr.Dropdown(
                        label="Plot Type",
                        choices=["Scatter", "Line", "Bar", "Pie", "Box", "Histogram", "Heatmap"],
                        value="Scatter",
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    with gr.Row():
                        x_column_dropdown = gr.Dropdown(
                            label="X-Axis Column",
                            choices=[],
                            allow_custom_value=False,
                            interactive=True
                        )
                        
                        y_column_dropdown = gr.Dropdown(
                            label="Y-Axis Column (optional for some plots)",
                            choices=[],
                            allow_custom_value=False,
                            interactive=True
                        )
                    
                    aggregation_dropdown = gr.Dropdown(
                        label="Aggregation Method",
                        choices=["None", "Sum", "Mean", "Count", "Median", "Min", "Max"],
                        value="None",
                        allow_custom_value=False,
                        interactive=True
                    )
                    
                    gr.Markdown("""
                    **Aggregation Methods:**
                    - **None**: Plot raw data points
                    - **Sum**: Sum Y values for each unique X
                    - **Mean**: Average Y values for each unique X
                    - **Count**: Count occurrences for each unique X
                    - **Median/Min/Max**: Respective aggregation per X
                    """)
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Custom Labels (Optional)")
                    gr.Markdown("*Leave blank to use column names as defaults*")
                    
                    with gr.Row():
                        custom_x_label = gr.Textbox(
                            label="X-Axis Label",
                            placeholder="Enter custom x-axis label...",
                            interactive=True
                        )
                        
                        custom_y_label = gr.Textbox(
                            label="Y-Axis Label", 
                            placeholder="Enter custom y-axis label...",
                            interactive=True
                        )
                    
                    custom_title = gr.Textbox(
                        label="Chart Title",
                        placeholder="Enter custom title (default: 'X vs Y: Plot Type')",
                        interactive=True
                    )
            
            gr.Markdown("---")
            
            with gr.Row():
                generate_plot_btn = gr.Button(
                    "Generate Plot",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Row():
                plot_output = gr.Plot(
                    label="Visualization"
                )
                
            plot_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )
        
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
                    filter_file_dropdown,
                    viz_file_dropdown]
        )

        preview_btn.click(
            fn=dp.preview_file,
            inputs=[loaded_data, preview_file_dropdown],
            outputs=[preview_output]
        )

        # ==========================================
        # Statistics & Data Cleaning handlers
        # ==========================================
        
        # Toggle visibility of cleaning operation groups
        def toggle_cleaning_groups(op_type):
            return (
                gr.Group(visible=(op_type == "Convert Data Types")),
                gr.Group(visible=(op_type == "Fill Null Values")),
                gr.Group(visible=(op_type == "Drop Duplicates"))
            )
        
        cleaning_operation.change(
            fn=toggle_cleaning_groups,
            inputs=[cleaning_operation],
            outputs=[dtype_group, nulls_group, dupes_group]
        )
        
        # When dataset is selected, load all statistics and column info
        profile_file_dropdown.change(
            fn=utils.load_profile_and_columns,
            inputs=[loaded_data, profile_file_dropdown],
            outputs=[
                stats_summary, profile_output_summary, profile_output_columns,
                dtype_display, null_info_display, cleaning_preview,
                columns_to_convert, null_columns_dropdown, cleaning_log
            ]
        )
        
        # Convert data types
        convert_btn.click(
            fn=utils.convert_dtype_and_refresh,
            inputs=[loaded_data, profile_file_dropdown, columns_to_convert, target_dtype],
            outputs=[
                cleaning_log, loaded_data,
                stats_summary, profile_output_summary, profile_output_columns,
                dtype_display, null_info_display, cleaning_preview,
                columns_to_convert, null_columns_dropdown
            ]
        )
        
        # Fill null values
        fill_nulls_btn.click(
            fn=utils.fill_nulls_and_refresh,
            inputs=[loaded_data, profile_file_dropdown, null_columns_dropdown, fill_method_dropdown],
            outputs=[
                cleaning_log, loaded_data,
                stats_summary, profile_output_summary, profile_output_columns,
                dtype_display, null_info_display, cleaning_preview,
                columns_to_convert, null_columns_dropdown
            ]
        )
        
        # Drop duplicates
        drop_dupes_btn.click(
            fn=utils.drop_duplicates_and_refresh,
            inputs=[loaded_data, profile_file_dropdown],
            outputs=[
                cleaning_log, loaded_data,
                stats_summary, profile_output_summary, profile_output_columns,
                dtype_display, null_info_display, cleaning_preview,
                columns_to_convert, null_columns_dropdown
            ]
        )
        
        # Download
        download_btn.click(
            fn=utils.prepare_download,
            inputs=[loaded_data, profile_file_dropdown],
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
        
        # ==========================================
        # Visualization Handlers
        # ==========================================

        load_viz_columns_btn.click(
            fn=utils.update_viz_columns,
            inputs=[loaded_data, viz_file_dropdown],
            outputs=[x_column_dropdown, y_column_dropdown]
        )

        generate_plot_btn.click(
            fn=utils.generate_plot_wrapper,
            inputs=[loaded_data, viz_file_dropdown, plot_type_dropdown,
                    x_column_dropdown, y_column_dropdown,
                    custom_x_label, custom_y_label, custom_title,
                    aggregation_dropdown],
            outputs=[plot_output, plot_status]
        )

    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch()