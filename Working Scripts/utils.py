from pathlib import Path
import data_processor as dp

# Data Upload Tab 
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
        return "No valid files or directories selected.", {}

    # Summarize all files
    summary = dp.summarize_directory(paths)
    supported = summary["supported"]
    unsupported = summary["unsupported"]

    # Load supported files
    loaded_result = dp.load_files(paths[0])  # load_files expects a single folder/file
    loaded = list(loaded_result["loaded"].keys())
    failed = loaded_result["failed"]

    # Build status message
    status_msg = "### Data Upload Summary\n"
    status_msg += f"Supported files found: {supported}\n"
    status_msg += f"Unsupported files: {unsupported}\n"
    status_msg += f"Successfully loaded: {loaded}\n"
    status_msg += f"Failed to load: {failed}"

    # Return message + loaded data dict
    return status_msg, loaded_result["loaded"]

def profile_file(loaded_data, selected_file):
    """
    Takes the loaded_data dict (from gr.State) and the selected file name,
    returns a profile summary string.
    """
    if not loaded_data:
        return "No datasets loaded. Please upload first."
    
    if selected_file not in loaded_data:
        return f"Dataset '{selected_file}' not found in loaded data."
    
    df = loaded_data[selected_file]
    profile = dp.profile(df) 

    # Convert profile dict to Markdown-friendly string
    profile_md = f"### Profile of {selected_file}\n"
    profile_md += f"- Shape: {profile['shape']}\n"
    profile_md += f"- Nulls per column:\n"
    for col, null_count in profile["nulls"].items():
        profile_md += f"  - {col}: {null_count}\n"
    profile_md += f"- Duplicates: {profile['duplicates']}\n"
    profile_md += f"- Summary statistics:\n"
    for col, stats in profile["describe"].items():
        profile_md += f"  - {col}:\n"
        for stat_name, value in stats.items():
            profile_md += f"    - {stat_name}: {value}\n"
    
    return profile_md

def update_dropdown_choices(loaded_data):
    return list(loaded_data.keys())
