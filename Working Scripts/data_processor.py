import pandas as pd
import random
from pathlib import Path

# Loading Data
SUPPORTED_EXTENSIONS = {
    ".csv": pd.read_csv,
    ".txt": pd.read_csv,
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

def load_files(directory):
    directory = Path(directory)
    loaded, failed, unsupported = {}, [], []

    for file in directory.glob("*"):
        if not file.is_file():
            continue
        loader = SUPPORTED_EXTENSIONS.get(file.suffix)
        if loader is None:
            unsupported.append(file.name)
            continue
        try:
            df = loader(file)
            loaded[file.stem] = df
        except Exception:
            failed.append(file.name)
    return {"loaded": loaded, "failed": failed, "unsupported": unsupported}


# Profiling data
def head_tail(df, n=5):
    if df.shape[0] > 2 * n:
        return {"head": df.head(n), "tail": df.tail(n)}
    return {"full": df}

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


# Cleaning Data
def convert_dtype(df, columns, dtype):
    for col in columns:
        df[col] = df[col].astype(dtype)
    return df

def drop_columns(df, columns):
    return df.drop(columns=columns, errors="ignore")

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
    return df