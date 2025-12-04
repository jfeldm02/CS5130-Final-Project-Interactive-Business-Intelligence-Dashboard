from pathlib import Path
import data_processor as dp
import gradio as gr
import pandas as pd
import tempfile
import os
import insights
import visualizations as viz

#=========================================================================================
# Data Upload Tab
#=========================================================================================

def data_upload_pipeline(uploaded_paths):
    """
    Takes uploaded file(s) or directory from Gradio,
    loads the supported datasets, and returns a status message
    and the loaded data dictionary for other tabs.
    """
    # Ensure uploaded_paths is a list
    if isinstance(uploaded_paths, str):
        uploaded_paths = [uploaded_paths]

    # Flatten paths and filter valid files/directories
    paths = [Path(p) for p in uploaded_paths if Path(p).exists()]

    if not paths:
        return ("No valid files or directories selected.",
                {},
                gr.Dropdown(choices=[]),
                gr.Dropdown(choices=[]),
                gr.Dropdown(choices=[]),
                gr.Dropdown(choices=[])
            )

    # Summarize all files
    summary = dp.summarize_directory(paths)
    supported = summary["supported"]
    unsupported = summary["unsupported"]

    # Load supported files
    loaded_result = dp.load_files(paths)  # load_files expects a single folder/file
    loaded = list(loaded_result["loaded"].keys())
    failed = loaded_result["failed"]

    # Build status message
    status_msg = "### Data Upload Summary\n"
    status_msg += f"Supported files found: {supported}\n"
    status_msg += f"Unsupported files: {unsupported}\n"
    status_msg += f"Successfully loaded: {loaded}\n"
    status_msg += f"Failed to load: {failed}"

    # Return message + loaded data dict + dropdowns for:
    # preview_file_dropdown, profile_file_dropdown, filter_file_dropdown
    return (status_msg, loaded_result["loaded"],
            gr.Dropdown(choices=loaded),
            gr.Dropdown(choices=loaded),
            gr.Dropdown(choices=loaded),
            gr.Dropdown(choices=[])
            )

def profile_file(loaded_data, selected_file):
    """Returns two DataFrames: overall summary and per-column statistics."""
    if not loaded_data:
        return None, None
    
    if selected_file not in loaded_data:
        return None, None
    
    df = loaded_data[selected_file]
    profile = dp.profile(df)
    
    # Overall summary DataFrame (no nulls here)
    summary_data = {
        "Metric": ["Rows", "Columns", "Duplicates"],
        "Value": [profile["shape"][0], profile["shape"][1], profile["duplicates"]]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Per-column statistics DataFrame
    col_stats_data = []
    for col, stats in profile["describe"].items():
        row = {"Column": col}
        # Add null count first
        row["Nulls"] = profile["nulls"].get(col, 0)
        # Then add all other statistics
        for stat_name, value in stats.items():
            if isinstance(value, (int, float)):
                row[stat_name] = round(value, 2)
            else:
                row[stat_name] = value
        col_stats_data.append(row)
    
    col_stats_df = pd.DataFrame(col_stats_data)
    
    return summary_df, col_stats_df

def update_dropdown_choices(loaded_data):
    return list(loaded_data.keys())

#=========================================================================================
# Statistics and Data Cleaning Tab
#=========================================================================================

def get_column_dtypes(loaded_data, selected_file):
    """Returns DataFrame showing column names and their current data types."""
    if not loaded_data or selected_file not in loaded_data:
        return None
    
    df = loaded_data[selected_file]
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Current Data Type": df.dtypes.astype(str)
    })
    
    return dtype_df

def convert_dtype_wrapper(loaded_data, selected_file, columns, new_dtype):
    """Convert selected columns to new data type and return updated dtype info."""
    if not loaded_data or selected_file not in loaded_data:
        return "No dataset selected.", None, loaded_data
    
    if not columns:
        return "No columns selected.", None, loaded_data
    
    try:
        df = loaded_data[selected_file]
        df = dp.convert_dtype(df, columns, new_dtype)
        loaded_data[selected_file] = df
        
        # Return success message and updated dtype display
        updated_dtypes = get_column_dtypes(loaded_data, selected_file)
        message = f"Successfully converted {len(columns)} column(s) to {new_dtype}"
        
        return message, updated_dtypes, loaded_data
    
    except Exception as e:
        return f"Error converting data type: {str(e)}", None, loaded_data

def update_dtype_view_and_columns(loaded_data, selected_file):
        """Update both dtype display and column choices."""
        dtype_df = get_column_dtypes(loaded_data, selected_file)
        if dtype_df is not None:
            column_choices = dtype_df["Column"].tolist()
            return dtype_df, gr.Dropdown(choices=column_choices)
        return None, gr.Dropdown(choices=[])

def drop_duplicates_wrapper(loaded_data, selected_file):
    """Drop duplicates from dataset."""
    if not loaded_data or selected_file not in loaded_data:
        return "No dataset selected.", None, None, loaded_data
    
    try:
        df = loaded_data[selected_file]
        original_rows = len(df)
        
        # Drop duplicates in place
        df = dp.drop_duplicates(df)
        loaded_data[selected_file] = df
        
        rows_removed = original_rows - len(df)
        message = f"Removed {rows_removed} duplicate row(s) from '{selected_file}'\n{original_rows} rows → {len(df)} rows"
        
        # Return updated summary
        summary_df, col_stats_df = profile_file(loaded_data, selected_file)
        
        return message, summary_df, col_stats_df, loaded_data
    
    except Exception as e:
        return f"Error dropping duplicates: {str(e)}", None, None, loaded_data
    
def fill_nulls_wrapper(loaded_data, selected_file, columns, method):
    """Fill null values in selected columns using specified method."""
    if not loaded_data or selected_file not in loaded_data:
        return "No dataset selected.", None, None, loaded_data
    
    if not columns:
        return "No columns selected.", None, None, loaded_data
    
    try:
        df = loaded_data[selected_file]
        
        # Count nulls before
        nulls_before = {col: df[col].isnull().sum() for col in columns}
        total_nulls_before = sum(nulls_before.values())
        
        # Fill nulls
        df = dp.fill_nulls(df, columns, method)
        loaded_data[selected_file] = df
        
        # Count nulls after
        nulls_after = {col: df[col].isnull().sum() for col in columns}
        total_nulls_after = sum(nulls_after.values())
        
        # Build message
        message = f"Filled {total_nulls_before - total_nulls_after} null values using '{method}' method\n\n"
        message += "Nulls Remaining Per column:\n"
        for col in columns:
            message += f"  - {col}: {nulls_before[col]} → {nulls_after[col]}\n"
        
        return message, loaded_data
    
    except Exception as e:
        return f"Error filling nulls: {str(e)}", loaded_data


def get_columns_with_nulls(loaded_data, selected_file):
    """Get DataFrame of columns with null values and their counts."""
    if not loaded_data or selected_file not in loaded_data:
        return None, gr.Dropdown(choices=[])
    
    df = loaded_data[selected_file]
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    
    if len(columns_with_nulls) == 0:
        return pd.DataFrame({"Message": ["No null values found in dataset"]}), gr.Dropdown(choices=[])
    
    null_df = pd.DataFrame({
        "Column": columns_with_nulls.index,
        "Null Count": columns_with_nulls.values,
        "Percentage (Null Data/Total Data)": (columns_with_nulls.values / len(df) * 100).round(2)
    })
    
    return null_df, gr.Dropdown(choices=columns_with_nulls.index.tolist())

def prepare_download(loaded_data, selected_file):
    """Prepare cleaned dataset for download."""
    if not loaded_data or selected_file not in loaded_data:
        return None
    
    df = loaded_data[selected_file]
    
    # Create filename with _cleaned suffix
    if selected_file.endswith('.csv'):
        output_filename = selected_file.replace('.csv', '_cleaned.csv')
    elif selected_file.endswith('.xlsx'):
        output_filename = selected_file.replace('.xlsx', '_cleaned.xlsx')
    elif selected_file.endswith('.json'):
        output_filename = selected_file.replace('.json', '_cleaned.json')
    else:
        # Default to CSV if unknown extension
        output_filename = f"{selected_file}_cleaned.csv"
    
    # I didn't know about this, AI recommended it.
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, output_filename)
    
    # Save based on file type
    if output_filename.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_filename.endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    elif output_filename.endswith('.json'):
        df.to_json(output_path, orient='records', indent=2)
    
    return output_path

#=========================================================================================
# Statistics and Data Cleaning Tab - Consolidated Version
#=========================================================================================

def load_profile_and_columns(loaded_data, selected_file):
    """
    Load all profile information when a dataset is selected.
    Returns all the statistics displays and populates column dropdowns.
    """
    if not loaded_data or selected_file not in loaded_data:
        return (
            "*Select a dataset to view statistics*",
            None, None, None, None, None,
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[]),
            "Select a dataset to begin"
        )
    
    df = loaded_data[selected_file]
    profile = dp.profile(df)
    
    # Stats summary markdown
    stats_summary = f"**{selected_file}**: {profile['shape'][0]:,} rows × {profile['shape'][1]} columns | {profile['duplicates']} duplicates"
    
    # Overall summary DataFrame
    summary_data = {
        "Metric": ["Rows", "Columns", "Duplicates"],
        "Value": [profile["shape"][0], profile["shape"][1], profile["duplicates"]]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Per-column statistics DataFrame
    col_stats_data = []
    for col, stats in profile["describe"].items():
        row = {"Column": col, "Nulls": profile["nulls"].get(col, 0)}
        for stat_name, value in stats.items():
            if isinstance(value, (int, float)):
                row[stat_name] = round(value, 2)
            else:
                row[stat_name] = value
        col_stats_data.append(row)
    col_stats_df = pd.DataFrame(col_stats_data)
    
    # Data types DataFrame
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })
    
    # Null info DataFrame
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if len(columns_with_nulls) == 0:
        null_df = pd.DataFrame({"Status": ["No null values found"]})
        null_column_choices = []
    else:
        null_df = pd.DataFrame({
            "Column": columns_with_nulls.index,
            "Null Count": columns_with_nulls.values,
            "% Null": (columns_with_nulls.values / len(df) * 100).round(2)
        })
        null_column_choices = columns_with_nulls.index.tolist()
    
    # Preview
    preview_df = df.head(10)
    
    # Column choices for dropdowns
    all_columns = df.columns.tolist()
    
    return (
        stats_summary,
        summary_df,
        col_stats_df,
        dtype_df,
        null_df,
        preview_df,
        gr.Dropdown(choices=all_columns),
        gr.Dropdown(choices=null_column_choices),
        "Ready for cleaning operations"
    )

def _refresh_all_stats(loaded_data, selected_file):
    """Helper to refresh all statistics displays after a cleaning operation."""
    return load_profile_and_columns(loaded_data, selected_file)

def convert_dtype_and_refresh(loaded_data, selected_file, columns, new_dtype):
    """Convert data types and refresh all statistics."""
    if not loaded_data or selected_file not in loaded_data:
        return ("No dataset selected.", loaded_data,
                "*Select a dataset*", None, None, None, None, None,
                gr.Dropdown(choices=[]), gr.Dropdown(choices=[]))
    
    if not columns:
        stats = _refresh_all_stats(loaded_data, selected_file)
        return ("No columns selected.",) + (loaded_data,) + stats
    
    try:
        df = loaded_data[selected_file]
        original_dtypes = {col: str(df[col].dtype) for col in columns}
        
        df = dp.convert_dtype(df, columns, new_dtype)
        loaded_data[selected_file] = df
        
        # Build log message
        log_lines = [f"✓ Converted {len(columns)} column(s) to {new_dtype}:"]
        for col in columns:
            log_lines.append(f"  • {col}: {original_dtypes[col]} → {new_dtype}")
        log_msg = "\n".join(log_lines)
        
        stats = _refresh_all_stats(loaded_data, selected_file)
        return (log_msg,) + (loaded_data,) + stats[:-1] + (stats[-1],)
    
    except Exception as e:
        stats = _refresh_all_stats(loaded_data, selected_file)
        return (f"Error: {str(e)}",) + (loaded_data,) + stats[:-1] + (stats[-1],)

def fill_nulls_and_refresh(loaded_data, selected_file, columns, method):
    """Fill null values and refresh all statistics."""
    if not loaded_data or selected_file not in loaded_data:
        return ("No dataset selected.", loaded_data,
                "*Select a dataset*", None, None, None, None, None,
                gr.Dropdown(choices=[]), gr.Dropdown(choices=[]))
    
    if not columns:
        stats = _refresh_all_stats(loaded_data, selected_file)
        return ("No columns selected.",) + (loaded_data,) + stats
    
    try:
        df = loaded_data[selected_file]
        
        # Count nulls before
        nulls_before = {col: df[col].isnull().sum() for col in columns}
        total_before = sum(nulls_before.values())
        
        df = dp.fill_nulls(df, columns, method)
        loaded_data[selected_file] = df
        
        # Count nulls after
        nulls_after = {col: df[col].isnull().sum() for col in columns}
        total_after = sum(nulls_after.values())
        
        # Build log message
        log_lines = [f"✓ Filled {total_before - total_after} null values using '{method}':"]
        for col in columns:
            log_lines.append(f"  • {col}: {nulls_before[col]} → {nulls_after[col]}")
        log_msg = "\n".join(log_lines)
        
        stats = _refresh_all_stats(loaded_data, selected_file)
        return (log_msg,) + (loaded_data,) + stats[:-1] + (stats[-1],)
    
    except Exception as e:
        stats = _refresh_all_stats(loaded_data, selected_file)
        return (f"Error: {str(e)}",) + (loaded_data,) + stats[:-1] + (stats[-1],)

def drop_duplicates_and_refresh(loaded_data, selected_file):
    """Drop duplicates and refresh all statistics."""
    if not loaded_data or selected_file not in loaded_data:
        return ("No dataset selected.", loaded_data,
                "*Select a dataset*", None, None, None, None, None,
                gr.Dropdown(choices=[]), gr.Dropdown(choices=[]))
    
    try:
        df = loaded_data[selected_file]
        original_rows = len(df)
        
        df = dp.drop_duplicates(df)
        loaded_data[selected_file] = df
        
        rows_removed = original_rows - len(df)
        log_msg = f"✓ Removed {rows_removed} duplicate row(s)\n  {original_rows:,} → {len(df):,} rows"
        
        stats = _refresh_all_stats(loaded_data, selected_file)
        return (log_msg,) + (loaded_data,) + stats[:-1] + (stats[-1],)
    
    except Exception as e:
        stats = _refresh_all_stats(loaded_data, selected_file)
        return (f"Error: {str(e)}",) + (loaded_data,) + stats[:-1] + (stats[-1],)

def get_column_info(loaded_data, selected_file):
    """Get column names and types for building filters."""
    if not loaded_data or selected_file not in loaded_data:
        return [], {}
    
    df = loaded_data[selected_file]
    columns = df.columns.tolist()
    
    # Get info about each column
    column_info = {}
    for col in columns:
        column_info[col] = {
            'dtype': str(df[col].dtype),
            'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
            'unique_values': df[col].nunique(),
            'sample_values': df[col].dropna().unique()[:10].tolist()
        }
    
    return columns, column_info

def load_unique_values(loaded_data, selected_file, column):
    """Load unique values for a given column."""
    if not loaded_data or selected_file not in loaded_data or not column:
        return gr.Dropdown(choices=[])
    
    df = loaded_data[selected_file]
    unique_vals = df[column].dropna().unique().tolist()
    if len(unique_vals) > 100:
        unique_vals = unique_vals[:100]
    
    return gr.Dropdown(choices=[str(v) for v in unique_vals])

def load_filter_columns_and_preview(loaded_data, selected_file, pending_operations):
    """
    When dataset is selected, load columns into all dropdowns,
    show initial preview, and reset pending operations.
    """
    if not loaded_data or selected_file not in loaded_data:
        empty_dropdown = gr.Dropdown(choices=[])
        return (
            empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown, empty_dropdown,
            None, "*Select a dataset to begin*", [], "No operations added yet"
        )
    
    columns, _ = get_column_info(loaded_data, selected_file)
    df = loaded_data[selected_file]
    
    dropdown_update = gr.Dropdown(choices=columns)
    preview_df = df.head(20)
    stats = f"**Original:** {len(df)} rows × {len(df.columns)} columns"
    
    return (
        dropdown_update, dropdown_update, dropdown_update, dropdown_update, dropdown_update,
        preview_df, stats, [], "No operations added yet"
    )

def apply_operations_to_df(df, operations):
    """
    Apply a list of operations to a DataFrame and return the result.
    Returns (transformed_df, operation_logs)
    """
    result_df = df.copy()
    logs = []
    
    for op in operations:
        op_type = op['type']
        
        if op_type == 'sort':
            result_df, log = insights.sort(
                result_df, 
                op['columns'], 
                op['ascending']
            )
            logs.append(log)
        
        elif op_type == 'filter_range':
            result_df, log = insights.filter_range(
                result_df,
                op['column'],
                op.get('min'),
                op.get('max')
            )
            logs.append(log)
        
        elif op_type == 'filter_values':
            result_df, log = insights.filter_values(
                result_df,
                op['column'],
                op['values']
            )
            logs.append(log)
        
        elif op_type == 'rename':
            result_df, log = insights.rename_columns(
                result_df,
                {op['old_name']: op['new_name']}
            )
            logs.append(log)
        
        elif op_type == 'select_columns':
            result_df, log = insights.select_columns(
                result_df,
                op['columns']
            )
            logs.append(log)
    
    return result_df, logs

def format_operations_summary(operations):
    """Format the list of operations into a readable summary."""
    if not operations:
        return "No operations added yet"
    
    lines = []
    for i, op in enumerate(operations, 1):
        op_type = op['type']
        
        if op_type == 'sort':
            order = "↑" if op['ascending'] else "↓"
            lines.append(f"{i}. Sort by {', '.join(op['columns'])} {order}")
        
        elif op_type == 'filter_range':
            min_val = op.get('min', '—')
            max_val = op.get('max', '—')
            lines.append(f"{i}. Filter {op['column']}: [{min_val} to {max_val}]")
        
        elif op_type == 'filter_values':
            vals = ', '.join(str(v) for v in op['values'][:3])
            if len(op['values']) > 3:
                vals += f" (+{len(op['values']) - 3} more)"
            lines.append(f"{i}. Keep {op['column']} in [{vals}]")
        
        elif op_type == 'rename':
            lines.append(f"{i}. Rename: {op['old_name']} → {op['new_name']}")
        
        elif op_type == 'select_columns':
            cols = ', '.join(op['columns'][:3])
            if len(op['columns']) > 3:
                cols += f" (+{len(op['columns']) - 3} more)"
            lines.append(f"{i}. Select columns: [{cols}]")
    
    return "\n".join(lines)

def add_operation(pending_operations, operation_type,
                  sort_columns, sort_order,
                  range_column, range_min, range_max,
                  values_column, available_values,
                  rename_old, rename_new,
                  select_columns,
                  loaded_data, selected_file):
    """
    Add a new operation to the pending list and update the live preview.
    """
    if not loaded_data or selected_file not in loaded_data:
        return pending_operations, "No dataset selected", None, "*Select a dataset*"
    
    # Build the operation based on type
    new_op = None
    
    if operation_type == "Sort":
        if sort_columns:
            new_op = {
                'type': 'sort',
                'columns': sort_columns,
                'ascending': sort_order == "Ascending"
            }
    
    elif operation_type == "Filter (Range)":
        if range_column and (range_min is not None or range_max is not None):
            new_op = {
                'type': 'filter_range',
                'column': range_column,
                'min': range_min,
                'max': range_max
            }
    
    elif operation_type == "Filter (Values)":
        if values_column and available_values:
            # Convert string values back to original types
            df = loaded_data[selected_file]
            original_dtype = df[values_column].dtype
            converted_values = available_values
            
            if pd.api.types.is_numeric_dtype(original_dtype):
                try:
                    converted_values = [
                        float(v) if '.' in str(v) else int(v) 
                        for v in available_values
                    ]
                except:
                    pass
            
            new_op = {
                'type': 'filter_values',
                'column': values_column,
                'values': converted_values
            }
    
    elif operation_type == "Rename Column":
        if rename_old and rename_new:
            new_op = {
                'type': 'rename',
                'old_name': rename_old,
                'new_name': rename_new
            }
    
    elif operation_type == "Select Columns":
        if select_columns:
            new_op = {
                'type': 'select_columns',
                'columns': select_columns
            }
    
    if new_op is None:
        # No valid operation configured
        summary = format_operations_summary(pending_operations)
        df = loaded_data[selected_file]
        transformed_df, _ = apply_operations_to_df(df, pending_operations)
        stats = f"**Original:** {len(df)} rows × {len(df.columns)} cols → **Preview:** {len(transformed_df)} rows × {len(transformed_df.columns)} cols"
        return pending_operations, summary, transformed_df.head(20), stats
    
    # Add the new operation
    updated_operations = pending_operations + [new_op]
    
    # Apply all operations and get preview
    df = loaded_data[selected_file]
    transformed_df, logs = apply_operations_to_df(df, updated_operations)
    
    summary = format_operations_summary(updated_operations)
    stats = f"**Original:** {len(df)} rows × {len(df.columns)} cols → **Preview:** {len(transformed_df)} rows × {len(transformed_df.columns)} cols"
    
    return updated_operations, summary, transformed_df.head(20), stats

def clear_operations(loaded_data, selected_file):
    """Clear all pending operations and reset preview."""
    if not loaded_data or selected_file not in loaded_data:
        return [], "No operations added yet", None, "*Select a dataset*"
    
    df = loaded_data[selected_file]
    stats = f"**Original:** {len(df)} rows × {len(df.columns)} columns"
    
    return [], "No operations added yet", df.head(20), stats

def undo_operation(pending_operations, loaded_data, selected_file):
    """Remove the last operation and update preview."""
    if not loaded_data or selected_file not in loaded_data:
        return [], "No operations added yet", None, "*Select a dataset*"
    
    df = loaded_data[selected_file]
    
    if not pending_operations:
        stats = f"**Original:** {len(df)} rows × {len(df.columns)} columns"
        return [], "No operations added yet", df.head(20), stats
    
    # Remove last operation
    updated_operations = pending_operations[:-1]
    
    # Apply remaining operations
    transformed_df, _ = apply_operations_to_df(df, updated_operations)
    
    summary = format_operations_summary(updated_operations)
    stats = f"**Original:** {len(df)} rows × {len(df.columns)} cols → **Preview:** {len(transformed_df)} rows × {len(transformed_df.columns)} cols"
    
    return updated_operations, summary, transformed_df.head(20), stats

def save_transformed_dataset(loaded_data, selected_file, pending_operations, save_name):
    """
    Apply all pending operations and save as a new dataset.
    """
    if not loaded_data or selected_file not in loaded_data:
        return loaded_data, [], "No operations added yet", None, "*Select a dataset*", gr.Dropdown()
    
    if not pending_operations:
        df = loaded_data[selected_file]
        stats = f"**Original:** {len(df)} rows × {len(df.columns)} columns"
        return loaded_data, [], "No operations to apply", df.head(20), stats, gr.Dropdown(choices=list(loaded_data.keys()))
    
    df = loaded_data[selected_file]
    transformed_df, logs = apply_operations_to_df(df, pending_operations)
    
    # Determine save name
    if save_name and save_name.strip():
        new_name = save_name.strip()
        if new_name in loaded_data:
            # Add suffix if name exists
            counter = 1
            base_name = new_name
            while new_name in loaded_data:
                new_name = f"{base_name}_{counter}"
                counter += 1
    else:
        new_name = f"{selected_file}_transformed"
        counter = 1
        while new_name in loaded_data:
            new_name = f"{selected_file}_transformed_{counter}"
            counter += 1
    
    # Save the transformed dataset
    loaded_data[new_name] = transformed_df
    
    # Build success message
    summary = f"✓ Saved as '{new_name}'\n\nApplied {len(pending_operations)} operation(s):\n"
    summary += "\n".join(f"  • {log}" for log in logs)
    
    stats = f"**Saved:** {len(transformed_df)} rows × {len(transformed_df.columns)} columns"
    
    # Update dropdown with new dataset
    updated_dropdown = gr.Dropdown(choices=list(loaded_data.keys()), value=new_name)
    
    return loaded_data, [], summary, transformed_df.head(20), stats, updated_dropdown

#=========================================================================================
# Visualizations Tab
#=========================================================================================

def update_viz_columns(loaded_data, selected_file):
    """Update column dropdowns for visualization tab."""
    if not loaded_data or selected_file not in loaded_data:
        return gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
    
    df = loaded_data[selected_file]
    columns = df.columns.tolist()
    
    return gr.Dropdown(choices=columns), gr.Dropdown(choices=columns)


def generate_plot_wrapper(loaded_data, selected_file, plot_type,
                          x_col, y_col, x_label, y_label, title,
                          aggregation):
    """Wrapper to generate plot from Gradio inputs."""
    if not loaded_data or selected_file not in loaded_data:
        return None, "No dataset selected."
    
    if not x_col:
        return None, "Please select an X-axis column."
    
    df = loaded_data[selected_file]
    
    # Validate columns exist
    if x_col not in df.columns:
        return None, f"Column '{x_col}' not found in dataset."
    
    if y_col and y_col not in df.columns:
        return None, f"Column '{y_col}' not found in dataset."
    
    try:
        fig = viz.generate_plot(
            plot_type=plot_type,
            data=df,
            x_col=x_col,
            y_col=y_col if y_col else None,
            x_label=x_label,
            y_label=y_label,
            title=title,
            aggregation=aggregation
        )
        
        if fig is None:
            return None, f"Could not generate {plot_type} plot. Check that you've selected appropriate columns."
        
        status = f"Generated {plot_type} plot"
        if aggregation != "None":
            status += f" with {aggregation} aggregation"
        
        return fig, status
        
    except Exception as e:
        return None, f"Error generating plot: {str(e)}"