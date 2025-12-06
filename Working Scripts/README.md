# Interactive Business Intelligence Dashboard

**CS5130: Final Project**  
**Author:** Justin Feldman  
**Instructor:** Professor Lino Coria Mendoza

## Overview

A comprehensive business intelligence dashboard built with Python and Gradio that enables users to upload datasets, perform data cleaning and transformation operations, generate interactive visualizations, and automatically discover insights from their data.

## Features

### Data Upload
- Support for multiple file formats: CSV, TSV, TXT, Excel (.xlsx, .xls), JSON, Parquet, Feather, and HDF5
- Multi-file upload capability
- Automatic file type detection and loading
- Head/tail preview of loaded datasets

### Statistics & Data Cleaning
- Dataset profiling with row/column counts, duplicate detection, and null value analysis
- Column-wise statistics (mean, median, std, min, max, quartiles)
- Data type conversion across columns
- Null value handling with multiple strategies (mean, median, mode, random fill, remove)
- Duplicate row removal
- Export cleaned datasets to CSV, Excel, or JSON

### Filter & Explore
- Interactive operation builder with live preview
- Sort by single or multiple columns (ascending/descending)
- Range filtering for numeric columns
- Value filtering with multi-select
- Date range filtering
- Column renaming
- Column selection/projection
- Pending operations system with undo capability
- Save transformed datasets as new entries

### Visualizations
- Seven plot types: Scatter, Line, Bar, Pie, Box, Histogram, Heatmap
- Built with Plotly for full interactivity (zoom, pan, hover)
- Aggregation options: Sum, Mean, Count, Median, Min, Max
- Custom axis labels and chart titles
- Export to PNG, JPG, SVG, PDF, or interactive HTML

### Automated Insights
- Top/bottom performer identification
- Trend detection using linear regression
- Anomaly detection with configurable σ threshold
- Distribution statistics (skewness, IQR, quartiles)
- Quick scan across all numeric columns

## Installation

1. Clone or download the project files
2. Create and activate a conda environment (recommended):
   ```bash
   conda create -n bi-dashboard python=3.10
   conda activate bi-dashboard
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Launch the dashboard:
```bash
python app.py
```

The application will start a local server and open in your default browser (typically at `http://127.0.0.1:7860`).

### Workflow

1. **Upload Data** — Use the Data Upload tab to load one or more datasets
2. **Clean Data** — Switch to Statistics & Data Cleaning to profile and clean your data
3. **Transform** — Use Filter & Explore to sort, filter, rename, and reshape your data
4. **Visualize** — Create interactive charts in the Visualizations tab
5. **Discover** — Generate automated insights in the Insights tab

## Project Structure

```
├── app.py              # Main Gradio application and UI layout
├── data_processor.py   # Data loading, profiling, and cleaning functions
├── insights.py         # Filter operations and automated insight generation
├── visualizations.py   # Plotly chart generation
├── utils.py            # Utility functions and Gradio event handlers
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Supported File Formats

| Extension | Format |
|-----------|--------|
| .csv | Comma-separated values |
| .tsv | Tab-separated values |
| .txt | Auto-detected delimiter |
| .xlsx | Excel (modern) |
| .xls | Excel (legacy) |
| .json | JSON |
| .parquet | Apache Parquet |
| .feather | Apache Feather |
| .h5, .hdf5 | HDF5 |

## Dependencies

- **Gradio** — Web interface framework
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Plotly** — Interactive visualizations
- **Kaleido** — Static image export for Plotly
- **openpyxl / xlrd** — Excel file support
- **pyarrow** — Parquet and Feather support
- **tables** — HDF5 support

## License

This project was developed for academic purposes as part of CS5130.
