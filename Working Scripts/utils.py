from pathlib import Path
import data_processor as dp
import gradio as gr
import pandas as pd
import tempfile
import os
import insights

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

    # Return message + loaded data dict
    return (status_msg, loaded_result["loaded"],
            gr.Dropdown(choices=loaded),
            gr.Dropdown(choices=loaded),
            gr.Dropdown(choices=loaded),
            gr.Dropdown(choices=loaded),
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
        return f"Error filling nulls: {str(e)}", None, None, loaded_data


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
# Filter and Explore Tab
#=========================================================================================

def operations(loaded_data, selected_file, operations, save_name=None, preview_only=False):
    """
    Apply multiple operations to a dataset.
    
    operations is a list of dicts with structure:
    {
        'type': 'sort' | 'filter_range' | 'filter_values' | 'sum' | 'select_columns' | 'rename_columns',
        'columns': [list of column names],
        'params': {additional parameters based on operation type}
    }
    """
    if not loaded_data or selected_file not in loaded_data:
        return "No dataset selected.", None, loaded_data
    
    try:
        df = loaded_data[selected_file].copy()
        operation_log = []
        
        for op in operations:
            op_type = op['type']
            columns = op.get('columns', [])
            params = op.get('params', {})
            
            if op_type == 'sort':
                df, log = insights.sort(df, columns, params.get('ascending', True))
                operation_log.append(log)
            
            elif op_type == 'filter_range':
                df, log = insights.filter_range(
                    df, 
                    columns[0], 
                    params.get('min'), 
                    params.get('max')
                )
                operation_log.append(log)
            
            elif op_type == 'filter_values':
                df, log = insights.filter_values(
                    df, 
                    columns[0], 
                    params.get('values', [])
                )
                operation_log.append(log)
            
            elif op_type == 'sum':
                df, log = sum(df, columns)
                operation_log.append(log)
            
            elif op_type == 'select_columns':
                df, log = insights.select_columns(df, columns)
                operation_log.append(log)
            
            elif op_type == 'rename_columns':
                df, log = insights.rename_columns(df, params.get('rename_map', {}))
                operation_log.append(log)
            
            else:
                operation_log.append(f"Unknown operation: {op_type}")
        
        if preview_only:
            message = f"Preview: {len(df)} rows and {len(df.columns)} columns\n\n"
            message += "Operations that will be applied:\n" + "\n".join(f"  - {log}" for log in operation_log)
            message += "\n\n⚠️ This is a preview only. Click 'Apply Operations & Save' to save these changes."
            return message, df.head(20), loaded_data

        if save_name and save_name.strip():
            # User provided a custom name
            new_name = save_name.strip()
            
            # Check if name already exists
            if new_name in loaded_data:
                return f"Error: A dataset named '{new_name}' already exists. Choose a different name.", None, loaded_data
        else:
            # Use default naming with _transformed suffix
            new_name = f"{selected_file}_transformed"
            counter = 1
            while new_name in loaded_data:
                new_name = f"{selected_file}_transformed_{counter}"
                counter += 1
        
        # Save the transformed dataset
        loaded_data[new_name] = df
        
        message = f"Created '{new_name}' with {len(df)} rows and {len(df.columns)} columns\n\n"
        message += "Operations applied:\n" + "\n".join(f"  - {log}" for log in operation_log)
        
        return message, df.head(20), loaded_data
    
    except Exception as e:
        return f"Error applying operations: {str(e)}", None, loaded_data

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
 
    # Load columns when dataset is selected
def update_filter_columns(loaded_data, selected_file):
    columns, _ = get_column_info(loaded_data, selected_file)
    dropdown_update = gr.Dropdown(choices=columns)
    return [dropdown_update] * 5  # Update all 5 dropdowns

# Load available values for selected column
def load_unique_values(loaded_data, selected_file, column):
    if not loaded_data or selected_file not in loaded_data or not column:
        return gr.Dropdown(choices=[])
    
    df = loaded_data[selected_file]
    unique_vals = df[column].dropna().unique().tolist()
    if len(unique_vals) > 100:
        unique_vals = unique_vals[:100]
    
    return gr.Dropdown(choices=[str(v) for v in unique_vals])

# Add rename to queue
def add_to_rename_queue(rename_map, old_name, new_name):
    if not old_name or not new_name:
        return rename_map, "No renames queued" if not rename_map else "\n".join([f"{k} → {v}" for k, v in rename_map.items()])
    
    rename_map = rename_map.copy()
    rename_map[old_name] = new_name
    
    display = "\n".join([f"{k} → {v}" for k, v in rename_map.items()])
    return rename_map, display

# Clear rename queue
def clear_rename_queue():
    return {}, "No renames queued"

def apply_all_operations(loaded_data, selected_file, sort_cols, sort_ord, 
                        range_col, r_min, r_max, val_col, val_list, 
                        rename_map, sel_cols, custom_name, preview_only=False):
    """
    Wrapper function that builds operations list from Gradio inputs
    and calls the operations function.
    """
    operations_list = []
    
    # Build operations list
    if sort_cols:
        operations_list.append({
            'type': 'sort',
            'columns': sort_cols,
            'params': {'ascending': sort_ord == "Ascending"}
        })
    
    if range_col and (r_min is not None or r_max is not None):
        operations_list.append({
            'type': 'filter_range',
            'columns': [range_col],
            'params': {'min': r_min, 'max': r_max}
        })
    
    if val_col and val_list:
        # Convert string values back to original types if needed
        df = loaded_data[selected_file]
        original_dtype = df[val_col].dtype
        
        if pd.api.types.is_numeric_dtype(original_dtype):
            try:
                val_list = [float(v) if '.' in str(v) else int(v) for v in val_list]
            except:
                pass
        
        operations_list.append({
            'type': 'filter_values',
            'columns': [val_col],
            'params': {'values': val_list}
        })
    
    if rename_map:
        operations_list.append({
            'type': 'rename_columns',
            'columns': [],
            'params': {'rename_map': rename_map}
        })
    
    if sel_cols:
        operations_list.append({
            'type': 'select_columns',
            'columns': sel_cols,
            'params': {}
        })
    
    if not operations_list:
        return "No operations selected.", None, loaded_data
    
    # Call the main operations function
    return operations(loaded_data, selected_file, operations_list, save_name=custom_name, preview_only=preview_only)