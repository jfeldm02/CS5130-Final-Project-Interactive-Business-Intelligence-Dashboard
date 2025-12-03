from pathlib import Path
import data_processor as dp
import gradio as gr
import pandas as pd

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

    # Return message + loaded data dict
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
# Data Cleaning Tab
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
        message += "Per column:\n"
        for col in columns:
            message += f"  - {col}: {nulls_before[col]} → {nulls_after[col]}\n"
        
        # Return updated summary
        summary_df, col_stats_df = profile_file(loaded_data, selected_file)
        
        return message, summary_df, col_stats_df, loaded_data
    
    except Exception as e:
        return f"Error filling nulls: {str(e)}", None, None, loaded_data


def get_columns_with_nulls(loaded_data, selected_file):
    """Get DataFrame of columns with null values and their counts."""
    if not loaded_data or selected_file not in loaded_data:
        return None, []
    
    df = loaded_data[selected_file]
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    
    if len(columns_with_nulls) == 0:
        return pd.DataFrame({"Message": ["No null values found in dataset"]}), []
    
    null_df = pd.DataFrame({
        "Column": columns_with_nulls.index,
        "Null Count": columns_with_nulls.values,
        "Percentage": (columns_with_nulls.values / len(df) * 100).round(2)
    })
    
    # Return both the display DataFrame and the list of column names for dropdown
    return null_df, columns_with_nulls.index.tolist()