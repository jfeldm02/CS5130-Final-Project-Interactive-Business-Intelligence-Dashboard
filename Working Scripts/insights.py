import pandas as pd
import numpy as np

#=========================================================================================
# Filter and Explore Tab
#=========================================================================================

def sort(df, columns, ascending=True):
    """Sort dataframe by specified columns."""
    df = df.sort_values(by=columns, ascending=ascending)
    log = f"Sorted by {columns} ({'ascending' if ascending else 'descending'})"
    return df, log


def filter_range(df, column, min_val=None, max_val=None):
    """Filter dataframe by numeric range."""
    if min_val is not None:
        df = df[df[column] >= min_val]
    if max_val is not None:
        df = df[df[column] <= max_val]
    log = f"Filtered {column} to range [{min_val}, {max_val}]"
    return df, log


def filter_values(df, column, values):
    """Filter dataframe to keep only specified values."""
    df = df[df[column].isin(values)]
    log = f"Filtered {column} to values: {values}"
    return df, log


def sum(df, columns):
    """Calculate sum of specified columns (doesn't modify df)."""
    sum_result = df[columns].sum()
    log = f"Sum of {columns}: {sum_result.to_dict()}"
    return df, log


def select_columns(df, columns):
    """Select only specified columns."""
    df = df[columns]
    log = f"Selected columns: {columns}"
    return df, log


def rename_columns(df, rename_map):
    """Rename columns according to mapping."""
    df = df.rename(columns=rename_map)
    log = f"Renamed columns: {rename_map}"
    return df, log

def filter_date_range(df, column, start_date=None, end_date=None):
    """Filter dataframe by date range."""
    # Ensure column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors='coerce')
    
    if start_date is not None:
        df = df[df[column] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[column] <= pd.to_datetime(end_date)]
    
    log = f"Filtered {column} from {start_date or 'beginning'} to {end_date or 'end'}"
    return df, log


#=========================================================================================
# Insights Tab
#=========================================================================================

def get_performers(df, value_col, label_col=None, n=5):
    """
    Identify top and bottom performers by a numeric column.
    
    Returns dict with 'top' and 'bottom' DataFrames.
    """
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return None
    
    sorted_df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    
    if label_col and label_col in df.columns:
        cols_to_show = [label_col, value_col]
    else:
        cols_to_show = [value_col]
    
    top = sorted_df.head(n)[cols_to_show].copy()
    bottom = sorted_df.tail(n)[cols_to_show].copy()
    
    # Add rank column
    top.insert(0, 'Rank', range(1, len(top) + 1))
    bottom.insert(0, 'Rank', range(len(sorted_df) - len(bottom) + 1, len(sorted_df) + 1))
    
    return {"top": top, "bottom": bottom}


def detect_trend(df, value_col, date_col=None):
    """
    Detect overall trend direction using simple linear regression slope.
    
    If date_col is provided, sorts by date first.
    Returns dict with trend direction, slope, and statistics.
    """
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return None
    
    working_df = df.copy()
    
    # Sort by date if provided
    if date_col and date_col in df.columns:
        working_df = working_df.sort_values(date_col)
    
    values = working_df[value_col].dropna()
    
    if len(values) < 3:
        return {"trend": "insufficient data", "slope": None}
    
    x = np.arange(len(values))
    
    # Simple slope calculation using polyfit
    slope, intercept = np.polyfit(x, values, 1)
    
    # Normalize slope by mean to get percentage change per unit
    mean_val = values.mean()
    if mean_val != 0:
        normalized_slope = (slope / mean_val) * 100
    else:
        normalized_slope = 0
    
    # Determine trend based on normalized slope
    if normalized_slope > 1:
        trend = "increasing"
    elif normalized_slope < -1:
        trend = "decreasing"
    else:
        trend = "stable"
    
    # Calculate additional stats
    first_half = values.iloc[:len(values)//2].mean()
    second_half = values.iloc[len(values)//2:].mean()
    
    return {
        "trend": trend,
        "slope": round(slope, 4),
        "normalized_slope_pct": round(normalized_slope, 2),
        "start_avg": round(first_half, 2),
        "end_avg": round(second_half, 2),
        "change_pct": round(((second_half - first_half) / first_half) * 100, 2) if first_half != 0 else 0
    }


def find_anomalies(df, value_col, threshold=2.5):
    """
    Flag values that deviate significantly from the mean using Z-scores.
    
    Returns dict with anomaly count and DataFrames of high/low anomalies.
    """
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return None
    
    values = df[value_col].dropna()
    
    if len(values) < 3:
        return {"count": 0, "high": pd.DataFrame(), "low": pd.DataFrame()}
    
    mean = values.mean()
    std = values.std()
    
    if std == 0:
        return {"count": 0, "high": pd.DataFrame(), "low": pd.DataFrame()}
    
    working_df = df.copy()
    working_df['_z_score'] = (working_df[value_col] - mean) / std
    
    high_anomalies = working_df[working_df['_z_score'] > threshold].copy()
    low_anomalies = working_df[working_df['_z_score'] < -threshold].copy()
    
    # Clean up and format
    high_anomalies['Deviation'] = 'High (+' + high_anomalies['_z_score'].round(2).astype(str) + 'σ)'
    low_anomalies['Deviation'] = 'Low (' + low_anomalies['_z_score'].round(2).astype(str) + 'σ)'
    
    high_anomalies = high_anomalies.drop(columns=['_z_score'])
    low_anomalies = low_anomalies.drop(columns=['_z_score'])
    
    return {
        "count": len(high_anomalies) + len(low_anomalies),
        "high": high_anomalies,
        "low": low_anomalies,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "threshold": threshold
    }


def get_distribution_stats(df, value_col):
    """
    Calculate distribution statistics for a numeric column.
    """
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return None
    
    values = df[value_col].dropna()
    
    return {
        "count": len(values),
        "mean": round(values.mean(), 2),
        "median": round(values.median(), 2),
        "std": round(values.std(), 2),
        "min": round(values.min(), 2),
        "max": round(values.max(), 2),
        "q1": round(values.quantile(0.25), 2),
        "q3": round(values.quantile(0.75), 2),
        "iqr": round(values.quantile(0.75) - values.quantile(0.25), 2),
        "skew": round(values.skew(), 2) if len(values) > 2 else 0
    }


def generate_all_insights(df, value_col, label_col=None, date_col=None, n_performers=5, anomaly_threshold=2.5):
    """
    Generate all insights for a single numeric column.
    
    Returns a dict containing all insight results.
    """
    insights = {
        "column": value_col,
        "performers": get_performers(df, value_col, label_col, n_performers),
        "trend": detect_trend(df, value_col, date_col),
        "anomalies": find_anomalies(df, value_col, anomaly_threshold),
        "distribution": get_distribution_stats(df, value_col)
    }
    
    return insights


def get_numeric_columns(df):
    """Return list of numeric column names."""
    return df.select_dtypes(include='number').columns.tolist()


def get_categorical_columns(df):
    """Return list of potential label/category columns (non-numeric with reasonable cardinality)."""
    categorical = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Include if cardinality is reasonable (not too high, not too low)
            unique_count = df[col].nunique()
            if 1 < unique_count <= len(df) * 0.5:
                categorical.append(col)
    return categorical


def get_date_columns(df):
    """Return list of datetime columns."""
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Also check for columns that might be dates stored as objects
    for col in df.columns:
        if col not in date_cols and df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(10))
                date_cols.append(col)
            except:
                pass
    
    return date_cols