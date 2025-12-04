import pandas as pd
import random
from pathlib import Path

#=========================================================================================
# Data Upload Tab
#=========================================================================================
SUPPORTED_EXTENSIONS = {
    ".csv": pd.read_csv,
    # Struggled with this line for generalized delimiters, AI helped
    ".txt": lambda f: pd.read_csv(f, sep=None, engine="python"), 
    ".tsv": lambda f: pd.read_csv(f, sep="\t"),
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".json": pd.read_json,
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
    ".h5": pd.read_hdf,
    ".hdf5": pd.read_hdf
}

def summarize_directory(files):
    files = [Path(f) for f in files]
    supported, unsupported = [], []
    for file in files:
        if file.is_file():
            if file.suffix in SUPPORTED_EXTENSIONS:
                supported.append(file.name)
            else:
                unsupported.append(file.name)
    return {"supported": supported, "unsupported": unsupported}

def load_files(files):
    loaded, failed, unsupported = {}, [], []

    # Accept a single file or a list of files
    if not isinstance(files, list):
        files = [files]

    for f in files:
        # Normalize file path (supports Path, str, and Gradio TemporaryUploadedFile)
        if hasattr(f, "path"):
            fpath = Path(f.path)
        else:
            fpath = Path(f)

        loader = SUPPORTED_EXTENSIONS.get(fpath.suffix)

        if loader is None:
            unsupported.append(fpath.name)
            continue

        try:
            df = loader(fpath)
            loaded[fpath.stem] = df
        except Exception:
            failed.append(fpath.name)

    return {"loaded": loaded, "failed": failed, "unsupported": unsupported}

def preview_file(loaded_data, selected_file, n=5):
    """
    Display head and tail of selected dataset.
    Returns a DataFrame showing first n and last n rows.
    """
    if not loaded_data:
        return None
    
    if selected_file not in loaded_data:
        return None
    
    df = loaded_data[selected_file]

    # If dataframe is big enough
    if df.shape[0] > 2 * n:
        result = {"head": df.head(n), "tail": df.tail(n)}
    else:
        result = {"full": df}
    
    if "full" in result:
        return result["full"]
    else:
        # Combine head and tail with a separator row
        separator = pd.DataFrame(
            [["..."] * len(df.columns)], 
            columns=df.columns,
            index=["..."]
        )
        return pd.concat([result["head"], separator, result["tail"]])

#=========================================================================================
# Statistics and Cleaning Tab
#=========================================================================================

def null_counts(df):
    return df.isnull().sum().to_dict()

def duplicates(df):
    return df.duplicated().sum()

def describe(df):
    return df.describe(include="all").to_dict()

def profile(df):
    return {
        "shape": df.shape,
        "nulls": null_counts(df),
        "duplicates": duplicates(df),
        "describe": describe(df),
    }

def convert_dtype(df, columns, dtype):
    for col in columns:
        df[col] = df[col].astype(dtype)
    return df

def drop_duplicates(df):
    return df.drop_duplicates()

def fill_nulls(df, columns, method="mean"):
    for col in columns:
        # AI made the recommendation for the pd.api.types.is_numeric_dtype check
        if method.lower() == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif method.lower() == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif method.lower() == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif method.lower() == "random":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(
                    df[col].apply(lambda _: random.uniform(df[col].min(), df[col].max()))
                )
            else:
                df[col] = df[col].fillna(lambda _: random.choice(df[col].dropna().unique()))
        elif method.lower() == "remove":
            df = df.dropna(subset=[col])

    return df