import plotly.express as px
import pandas as pd

PLOT_TYPES = ["Scatter", "Line", "Bar", "Pie", "Box", "Histogram", "Heatmap"]
AGGREGATION_METHODS = ["None", "Sum", "Mean", "Count", "Median", "Min", "Max"]


def aggregate_data(df, x_col, y_col, agg_method):
    """Aggregate data by x column using specified method on y column."""
    if agg_method == "None" or agg_method is None:
        return df
    
    agg_map = {
        "Sum": "sum",
        "Mean": "mean",
        "Count": "count",
        "Median": "median",
        "Min": "min",
        "Max": "max"
    }
    
    pandas_agg = agg_map.get(agg_method, "sum")
    
    if y_col and pd.api.types.is_numeric_dtype(df[y_col]):
        aggregated = df.groupby(x_col, as_index=False)[y_col].agg(pandas_agg)
    else:
        if agg_method == "Count":
            aggregated = df.groupby(x_col, as_index=False).size()
            aggregated.columns = [x_col, "count"]
        else:
            aggregated = df
    
    return aggregated


def generate_plot(plot_type, data, x_col, y_col=None, 
                  x_label=None, y_label=None, title=None,
                  aggregation="None"):
    """Generates various plot types with customizable labels and aggregation."""
    if data is None or x_col is None:
        return None
    
    plot_data = aggregate_data(data.copy(), x_col, y_col, aggregation)
    
    if aggregation == "Count" and "count" in plot_data.columns:
        y_col = "count"
    
    x_label = x_label if x_label and x_label.strip() else x_col
    y_label = y_label if y_label and y_label.strip() else (y_col if y_col else "Value")
    
    if not title or not title.strip():
        if y_col:
            if aggregation and aggregation != "None":
                title = f"{aggregation} of {y_col} by {x_col}: {plot_type} Plot"
            else:
                title = f"{x_col} vs {y_col}: {plot_type} Plot"
        else:
            if aggregation and aggregation != "None":
                title = f"{aggregation} of {x_col}: {plot_type} Plot"
            else:
                title = f"{x_col}: {plot_type} Plot"
    
    fig = None
    
    if plot_type == "Scatter":
        if y_col is None:
            return None
        fig = px.scatter(plot_data, x=x_col, y=y_col)
        
    elif plot_type == "Line":
        if y_col is None:
            return None
        fig = px.line(plot_data, x=x_col, y=y_col)
        
    elif plot_type == "Bar":
        if y_col is None:
            return None
        fig = px.bar(plot_data, x=x_col, y=y_col)
        
    elif plot_type == "Pie":
        if y_col:
            fig = px.pie(plot_data, names=x_col, values=y_col)
        else:
            counts = plot_data[x_col].value_counts()
            fig = px.pie(values=counts.values, names=counts.index)
            
    elif plot_type == "Box":
        if y_col:
            fig = px.box(plot_data, x=x_col, y=y_col)
        else:
            fig = px.box(plot_data, y=x_col)
            
    elif plot_type == "Histogram":
        fig = px.histogram(plot_data, x=x_col)
        
    elif plot_type == "Heatmap":
        if y_col is None:
            return None
        pivot = plot_data.pivot_table(
            index=x_col, 
            columns=y_col, 
            aggfunc='size', 
            fill_value=0
        )
        fig = px.imshow(pivot, aspect="auto")
    
    if fig is None:
        return None
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white"
    )
    
    return fig