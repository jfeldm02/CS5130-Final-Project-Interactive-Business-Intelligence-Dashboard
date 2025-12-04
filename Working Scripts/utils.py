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
            gr.Dropdown(choices=loaded),
            gr.Dropdown(choices=loaded)
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
            empty_dropdown, empty_dropdown, "*Select a dataset to begin*", [], "No operations added yet"
        )
    
    columns, _ = get_column_info(loaded_data, selected_file)
    df = loaded_data[selected_file]
    
    dropdown_update = gr.Dropdown(choices=columns)
    preview_df = df.head(20)
    stats = f"**Original:** {len(df)} rows × {len(df.columns)} columns"
    
    return (
        dropdown_update, dropdown_update, dropdown_update, 
        dropdown_update, dropdown_update, dropdown_update,
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
        columns = op.get('columns', [])
        params = op.get('params', {})
        
        if op_type == 'sort':
            ascending = params.get('ascending', True)
            result_df, log = insights.sort(result_df, columns, ascending)
            logs.append(log)
        
        elif op_type == 'filter_range':
            column = columns[0] if columns else op.get('column')
            result_df, log = insights.filter_range(
                result_df,
                column,
                params.get('min'),
                params.get('max')
            )
            logs.append(log)
        
        elif op_type == 'filter_values':
            column = columns[0] if columns else op.get('column')
            result_df, log = insights.filter_values(
                result_df,
                column,
                params.get('values', [])
            )
            logs.append(log)

        elif op_type == 'filter_date':
            column = op.get('column')
            start = op.get('start')
            end = op.get('end')
            # Check if filter_date_range exists, otherwise do manual filtering
            if hasattr(insights, 'filter_date_range'):
                result_df, log = insights.filter_date_range(result_df, column, start, end)
            else:
                # Manual date filtering
                if start:
                    result_df = result_df[result_df[column] >= start]
                if end:
                    result_df = result_df[result_df[column] <= end]
                log = f"Filtered {column} from {start or '...'} to {end or '...'}"
            logs.append(log)
        
        elif op_type == 'rename':
            old_name = columns[0] if columns else op.get('old_name')
            new_name = params.get('new_name', op.get('new_name'))
            result_df, log = insights.rename_columns(result_df, {old_name: new_name})
            logs.append(log)
        
        elif op_type == 'select':
            # Handle 'select' type (from add_operation)
            result_df, log = insights.select_columns(result_df, columns)
            logs.append(log)
        
        elif op_type == 'select_columns':
            # Handle 'select_columns' type (legacy)
            result_df, log = insights.select_columns(result_df, columns)
            logs.append(log)
        
        else:
            logs.append(f"Unknown operation: {op_type}")
    
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
                  date_column, start_date, end_date,
                  rename_old, rename_new,
                  select_columns,
                  loaded_data, selected_file):
    """Add an operation to the pending list and update preview."""
    
    if not loaded_data or selected_file not in loaded_data:
        return pending_operations, "No dataset selected", None, ""
    
    pending_operations = list(pending_operations)  # Copy
    new_op = None
    
    if operation_type == "Sort" and sort_columns:
        new_op = {
            'type': 'sort',
            'columns': sort_columns,
            'params': {'ascending': sort_order == "Ascending"},
            'display': f"Sort by {sort_columns} ({'asc' if sort_order == 'Ascending' else 'desc'})"
        }
    
    elif operation_type == "Filter (Range)" and range_column:
        if range_min is not None or range_max is not None:
            new_op = {
                'type': 'filter_range',
                'columns': [range_column],
                'params': {'min': range_min, 'max': range_max},
                'display': f"Filter {range_column}: [{range_min or '...'} to {range_max or '...'}]"
            }
    
    elif operation_type == "Filter (Values)" and values_column and available_values:
        new_op = {
            'type': 'filter_values',
            'columns': [values_column],
            'params': {'values': available_values},
            'display': f"Filter {values_column} to {len(available_values)} values"
        }
    
    elif operation_type == "Filter (Date)" and date_column:
        if start_date or end_date:
            new_op = {
                'type': 'filter_date',
                'column': date_column,
                'start': start_date if start_date else None,
                'end': end_date if end_date else None,
                'display': f"Filter {date_column}: [{start_date or '...'} to {end_date or '...'}]"
            }
    
    elif operation_type == "Rename Column" and rename_old and rename_new:
        new_op = {
            'type': 'rename',
            'columns': [rename_old],
            'params': {'new_name': rename_new},
            'display': f"Rename '{rename_old}' → '{rename_new}'"
        }
    
    elif operation_type == "Select Columns" and select_columns:
        new_op = {
            'type': 'select',
            'columns': select_columns,
            'params': {},
            'display': f"Keep {len(select_columns)} columns"
        }
    
    if new_op:
        pending_operations.append(new_op)
    
    # Apply operations to get preview
    df = loaded_data[selected_file].copy()
    df, _ = apply_operations_to_df(df, pending_operations)
    
    # Build summary
    if pending_operations:
        summary = "\n".join([f"{i+1}. {op['display']}" for i, op in enumerate(pending_operations)])
    else:
        summary = "No operations added yet"
    
    preview_stats = f"**Preview**: {len(df):,} rows × {len(df.columns)} columns"
    
    return pending_operations, summary, df.head(20), preview_stats

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
        return None, "No dataset selected.", None
    
    if not x_col:
        return None, "Please select an X-axis column.", None
    
    df = loaded_data[selected_file]
    
    if x_col not in df.columns:
        return None, f"Column '{x_col}' not found in dataset.", None
    
    if y_col and y_col not in df.columns:
        return None, f"Column '{y_col}' not found in dataset.", None
    
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
            return None, f"Could not generate {plot_type} plot. Check column selection.", None
        
        status = f"Generated {plot_type} plot"
        if aggregation != "None":
            status += f" with {aggregation} aggregation"
        
        # Return fig twice: once for display, once for state storage
        return fig, status, fig
        
    except Exception as e:
        return None, f"Error generating plot: {str(e)}", None
    
def save_plot(fig, filename, file_format):
    """Save the plot to a file for download."""
    if fig is None:
        return None
    
    if not filename or not filename.strip():
        filename = "plot"
    
    filename = filename.strip()
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"{filename}.{file_format}")
    
    try:
        if file_format == "html":
            fig.write_html(output_path)
        else:
            fig.write_image(output_path, format=file_format)
        
        return output_path
    except Exception as e:
        print(f"Error saving plot: {e}")
        return None
    
#=========================================================================================
# Insights Tab Utilities
#=========================================================================================

def load_insight_columns(loaded_data, selected_file):
    """
    Load column options for the insights tab dropdowns.
    Returns updates for value_col, label_col, and date_col dropdowns.
    """
    if not loaded_data or selected_file not in loaded_data:
        empty = gr.Dropdown(choices=[])
        return empty, empty, empty
    
    df = loaded_data[selected_file]
    
    numeric_cols = insights.get_numeric_columns(df)
    categorical_cols = insights.get_categorical_columns(df)
    date_cols = insights.get_date_columns(df)
    
    # Add "None" option for optional fields
    categorical_with_none = ["(None)"] + categorical_cols
    date_with_none = ["(None)"] + date_cols
    
    return (
        gr.Dropdown(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),
        gr.Dropdown(choices=categorical_with_none, value="(None)"),
        gr.Dropdown(choices=date_with_none, value="(None)")
    )


def generate_insights_wrapper(loaded_data, selected_file, value_col, label_col, date_col, n_performers, anomaly_threshold):
    """
    Generate all insights and format for Gradio display.
    Returns: summary_text, top_performers_df, bottom_performers_df, trend_df, anomalies_df, distribution_df
    """
    if not loaded_data or selected_file not in loaded_data:
        return "No dataset selected.", None, None, None, None, None
    
    if not value_col:
        return "Please select a numeric column to analyze.", None, None, None, None, None
    
    df = loaded_data[selected_file]
    
    # Handle "None" selections
    label = None if label_col == "(None)" else label_col
    date = None if date_col == "(None)" else date_col
    
    # Generate insights
    results = insights.generate_all_insights(
        df, 
        value_col, 
        label_col=label, 
        date_col=date, 
        n_performers=int(n_performers),
        anomaly_threshold=float(anomaly_threshold)
    )
    
    # Build summary text
    summary_parts = [f"## Insights for: `{value_col}`\n"]
    
    # Distribution summary
    dist = results["distribution"]
    if dist:
        summary_parts.append(f"**Distribution Overview**")
        summary_parts.append(f"- Count: {dist['count']:,} values")
        summary_parts.append(f"- Range: {dist['min']} to {dist['max']}")
        summary_parts.append(f"- Mean: {dist['mean']} | Median: {dist['median']}")
        summary_parts.append(f"- Std Dev: {dist['std']} | IQR: {dist['iqr']}")
        
        # Interpret skew
        skew = dist['skew']
        if skew > 1:
            skew_desc = "strongly right-skewed (long tail of high values)"
        elif skew > 0.5:
            skew_desc = "moderately right-skewed"
        elif skew < -1:
            skew_desc = "strongly left-skewed (long tail of low values)"
        elif skew < -0.5:
            skew_desc = "moderately left-skewed"
        else:
            skew_desc = "approximately symmetric"
        summary_parts.append(f"- Skewness: {skew} ({skew_desc})\n")
    
    # Trend summary
    trend = results["trend"]
    if trend and trend.get("trend") != "insufficient data":
        trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(trend["trend"], "")
        summary_parts.append(f"**Trend Analysis** {trend_emoji}")
        summary_parts.append(f"- Direction: **{trend['trend'].upper()}**")
        summary_parts.append(f"- Change: {trend['change_pct']}% (first half avg → second half avg)")
        summary_parts.append(f"- First half avg: {trend['start_avg']} | Second half avg: {trend['end_avg']}\n")
    
    # Anomalies summary
    anomalies = results["anomalies"]
    if anomalies:
        if anomalies["count"] > 0:
            summary_parts.append(f"**Anomalies Detected** ⚠️")
            summary_parts.append(f"- Found {anomalies['count']} outlier(s) beyond {anomalies['threshold']}σ")
            summary_parts.append(f"- High outliers: {len(anomalies['high'])}")
            summary_parts.append(f"- Low outliers: {len(anomalies['low'])}")
        else:
            summary_parts.append(f"**Anomalies** ✓")
            summary_parts.append(f"- No outliers detected beyond {anomalies['threshold']}σ threshold")
    
    summary_text = "\n".join(summary_parts)
    
    # Prepare DataFrames for display
    top_df = results["performers"]["top"] if results["performers"] else None
    bottom_df = results["performers"]["bottom"] if results["performers"] else None
    
    # Trend DataFrame
    if trend and trend.get("trend") != "insufficient data":
        trend_df = pd.DataFrame([{
            "Metric": "Trend Direction",
            "Value": trend["trend"].upper()
        }, {
            "Metric": "Slope (per row)",
            "Value": trend["slope"]
        }, {
            "Metric": "Normalized Slope (%)",
            "Value": f"{trend['normalized_slope_pct']}%"
        }, {
            "Metric": "First Half Average",
            "Value": trend["start_avg"]
        }, {
            "Metric": "Second Half Average",
            "Value": trend["end_avg"]
        }, {
            "Metric": "Overall Change",
            "Value": f"{trend['change_pct']}%"
        }])
    else:
        trend_df = pd.DataFrame([{"Metric": "Status", "Value": "Insufficient data for trend analysis"}])
    
    # Anomalies DataFrame
    if anomalies and anomalies["count"] > 0:
        anomalies_df = pd.concat([anomalies["high"], anomalies["low"]], ignore_index=True)
        # Limit to first 20 for display
        if len(anomalies_df) > 20:
            anomalies_df = anomalies_df.head(20)
    else:
        anomalies_df = pd.DataFrame([{"Status": "No anomalies detected"}])
    
    # Distribution DataFrame
    if dist:
        dist_df = pd.DataFrame([{
            "Statistic": k.replace("_", " ").title(),
            "Value": v
        } for k, v in dist.items()])
    else:
        dist_df = None
    
    return summary_text, top_df, bottom_df, trend_df, anomalies_df, dist_df


def quick_insights_all_columns(loaded_data, selected_file):
    """
    Generate a quick summary of insights for ALL numeric columns.
    Returns a summary DataFrame and detailed text.
    """
    if not loaded_data or selected_file not in loaded_data:
        return "No dataset selected.", None
    
    df = loaded_data[selected_file]
    numeric_cols = insights.get_numeric_columns(df)
    
    if not numeric_cols:
        return "No numeric columns found in dataset.", None
    
    summary_rows = []
    
    for col in numeric_cols:
        dist = insights.get_distribution_stats(df, col)
        trend = insights.detect_trend(df, col)
        anomalies = insights.find_anomalies(df, col)
        
        trend_indicator = ""
        if trend and trend.get("trend") != "insufficient data":
            trend_indicator = {"increasing": "↑", "decreasing": "↓", "stable": "→"}.get(trend["trend"], "")
        
        summary_rows.append({
            "Column": col,
            "Mean": dist["mean"] if dist else "-",
            "Median": dist["median"] if dist else "-",
            "Std Dev": dist["std"] if dist else "-",
            "Min": dist["min"] if dist else "-",
            "Max": dist["max"] if dist else "-",
            "Trend": f"{trend_indicator} {trend['change_pct']}%" if trend and trend.get("change_pct") else "-",
            "Anomalies": anomalies["count"] if anomalies else 0
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Build text summary
    text_parts = [f"## Quick Insights Summary\n"]
    text_parts.append(f"Analyzed **{len(numeric_cols)}** numeric columns in `{selected_file}`\n")
    
    # Highlight notable findings
    anomaly_cols = [r["Column"] for r in summary_rows if r["Anomalies"] > 0]
    if anomaly_cols:
        text_parts.append(f"⚠️ **Columns with anomalies:** {', '.join(anomaly_cols)}")
    
    increasing = [r["Column"] for r in summary_rows if "↑" in str(r["Trend"])]
    decreasing = [r["Column"] for r in summary_rows if "↓" in str(r["Trend"])]
    
    if increasing:
        text_parts.append(f"📈 **Increasing trends:** {', '.join(increasing)}")
    if decreasing:
        text_parts.append(f"📉 **Decreasing trends:** {', '.join(decreasing)}")
    
    return "\n".join(text_parts), summary_df