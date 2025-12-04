import pandas as pd
import numpy as np

#=========================================================================================
# Filter and Explore Tab
#=========================================================================================

def sort(df, columns, ascending=True):
    """
    Sorts df column.

    Inputs:
        df: selected dataframe
        columns: selected dataframe columns
        ascending: bool toggling ascending/descending sort
    
    Returns:
        df: same dataframe sorted by column and method
        log: string to trace df changes
    """
    df = df.sort_values(by=columns, ascending=ascending)
    log = f"Sorted by {columns} ({'ascending' if ascending else 'descending'})"
    return df, log


def filter_range(df, column, min_val=None, max_val=None):
    """
    Filters by range of df column.

    Inputs:
        df: selected dataframe
        columns: selected dataframe columns
        min_val: bottom range
        max_val: top range 
    
    Returns:
        df: same dataframe ranged by column and values
        log: string to trace df changes
    """
    if min_val is not None:
        df = df[df[column] >= min_val]
    if max_val is not None:
        df = df[df[column] <= max_val]
    log = f"Filtered {column} to range [{min_val}, {max_val}]"
    return df, log


def filter_values(df, column, values):
    """
    Filters by unique values of df column.

    Inputs:
        df: selected dataframe
        columns: selected dataframe columns
        values: selected unique values
    
    Returns:
        df: same dataframe filtered by column unique values
        log: string to trace df changes
    """
    df = df[df[column].isin(values)]
    log = f"Filtered {column} to values: {values}"
    return df, log


def sum(df, columns):
    """
    Sums df column.

    Inputs:
        df: selected dataframe
        columns: selected dataframe columns
    
    Returns:
        df: same dataframe with sum values
        log: string to trace df changes
    """
    sum_result = df[columns].sum()
    log = f"Sum of {columns}: {sum_result.to_dict()}"
    return df, log


def select_columns(df, columns):
    """
    Sorts df column.

    Inputs:
        df: selected dataframe
        columns: selected dataframe columns
    
    Returns:
        df: same dataframe with column masking
        log: string to trace df changes
    """
    df = df[columns]
    log = f"Selected columns: {columns}"
    return df, log


def rename_columns(df, rename_map):
    """
    Sorts df column.

    Inputs:
        df: selected dataframe
        rename_map: new name mapping
    
    Returns:
        df: same dataframe with new column names
        log: string to trace df changes
    """
    df = df.rename(columns=rename_map)
    log = f"Renamed columns: {rename_map}"
    return df, log

def filter_date_range(df, column, start_date=None, end_date=None):
    """
    Sorts df column by date range.

    Inputs:
        df: selected dataframe
        column: date column
        start_date: min range
        end_date: max range 
    
    Returns:
        df: same dataframe with date range filter
        log: string to trace df changes
    """
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
    Gets max/min values from specified column.

    Inputs:
        df: selected dataframe
        value_col: selected column in selected dataframe
        label_col: associated label column for context
        n: number of min/max values
    
    Returns:
        dictionary: two arrays, max and min values respectively 
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
    
    Inputs:
        df: selected dayaframe
        value_col: selected column of selected dataframe
        date_col: add date column for context
    
    Returns: 
        dictionary: trend direction, slope, and statistics.
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
    Flag outliers beyond a standard deviation.

    Inputs:
        df: selected dataframe
        value_col: selected column from selected dataframe
        threshold: standard deviation range from mean
    
    Returns:
        dictionary: anomaly count and dataframes of high/low outliers.
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

    Inputs:
        df: selected dataframe
        value_col: selected value column to analyze 
    
    Returns:
        dictionary: statistics 
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

    Inputs:
        df: selected dataframe
        value_col: column to analyze
        label_col: object column to supplement
        date_col: date column to supplement
        n_performers: number of preview data points
        anomaly_threshold: standard deviation range for outliers
    
    Returns:
        dictionary: containing all insight results
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
    """
    List of numeric column names.
    
    Inputs:
        df: selected dataframe
    
    Returns:
        List of numerical columns
    """
    return df.select_dtypes(include='number').columns.tolist()


def get_categorical_columns(df):
    """
    List of potential label/category columns.
    
    Inputs:
        df: dataframe
    
    Returns:
        categorical: List of categorical columns that are usable for suplementary info.
    """
    categorical = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Include if reasonable (not too high, not too low)
            unique_count = df[col].nunique()
            if 1 < unique_count <= len(df) * 0.5:
                categorical.append(col)
    return categorical


def get_date_columns(df):
    """
    List of datetime columns.
    
    Inputs: 
        df: selected dateframe
    
    Returns: 
        date_cols: list of columns with date-like structure 
    """
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