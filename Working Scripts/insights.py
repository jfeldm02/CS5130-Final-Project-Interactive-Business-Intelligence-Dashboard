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


