import pandas as pd
import random

def summarize_directory(files):
    """
    Describes the files input to the program. Details which files
    are and are not able to be processed by the program. 

    Parameters: 
        - DIRECTORY:    Global path of file/files

    Returns:
        - final_string: Summary of files uploaded   
    """

    output_string = []
    output_string.append(f"Data directory: {files}")
    output_string.append(f"\nFiles found:")

    supported_files = [
        ".csv", ".tsv", ".xlsx", 
        ".xls", ".json", ".h5",
        ".hdf5", ".parquet", ".feather"
    ]

    for file in files:
        if file.is_file() and not file.name.endswith('/'): # Check that it is a file and not a subfolder
            if file.name.endswith in supported_files:
                output_string.append(f"  - {file.name}")
            else:
                output_string.append(f"  - {file.name}     # WARNING: Unsupported file type! Will not be loaded.")
    
    final_string = "\n".join(output_string)

    return final_string

def upload_data(DIRECTORY):
    """
    Uploads the files provided, handling unsupported file types, 
    and giving summaries as to which files were successfully or 
    unsuccessfully loaded.

    Parameters: 
        - DIRECTORY:    Global path of file/files

    Returns:
        - data_dict:    Successfully loaded data labeled by file stem    
    """
    data_dict = {}
    bad_data_dict = []
    
    for file in DIRECTORY.glob('*'):
        if file.is_file():
            print(f"Loading {file.name}...")
            suffix = file.suffix
            filename = file.stem
            
            try:
                if suffix == '.csv':
                    df = pd.read_csv(file)
                    
                elif suffix == '.tsv':
                    df = pd.read_csv(file, sep='\t')
                
                elif suffix in ['.xlsx', '.xls']:
                    df = pd.read_excel(file)
                
                elif suffix == '.json':
                    df = pd.read_json(file)
                
                elif suffix == '.parquet':
                    df = pd.read_parquet(file)
                
                elif suffix == '.feather':
                    df = pd.read_feather(file)
                
                elif suffix in ['.h5', '.hdf5']:
                    df = pd.read_hdf(file)
                
                else:
                    print(f"  Skipping {file.name}. Unsupported file type.")
                    bad_data_dict.append(filename)
                    continue
                
                data_dict[filename] = df
                print(f"{filename} successfully loaded!")
                    
            except Exception as e:
                print(f"Error: Unable to load {filename}")
                bad_data_dict.append(filename)
        
        print("\n")
        print("="*50)
        print("\nSuccesfully Loaded:")
        for file in data_dict:
            print(f"     - {file}")
        
        print("\n")
        print("-"*50)

        print("\nFailed to Load:")
        for file in bad_data_dict:
            print(f"     - {file}")
        print("="*50)

    return data_dict

def data_head_tail(data_dict, file, number_rows= 5):
    """
    Shows the first and last desired rows of desired dataframe. 

    Parameters: 
        - data_dict:    Dictionary containing dataframe representations
                        of initial files
        
        - file:         Desired dataframe to visualize

    Returns:
        - Print statement of first desired rows of dataframe    
    """

    df = data_dict[file]

    if 2*number_rows < df.shape[0]:
        top_rows = df.head(number_rows)
        bottom_rows = df.tail(number_rows)
        print(f"\nTop {number_rows} of {file}:")
        print(top_rows)
        print("...")
        print(bottom_rows)
    else:
        print(f"\nOnly {df.shape[0]} rows in {file}:")
        print(df)

# I use this again later but only for this function so I made a standalone 
def null_per_column(df, file):
    # Null per column
    columns = df.columns

    print(f"{file} null values per column:")
    for column in columns:
        print(f'    - {column}:     {df[column].isnull().sum()}')

def dataframe_profiling(data_dict, file):
    """
    Profiles the desired dataframe with its info, basic statistics, and 
    null value count.  

    Parameters: 
        - data_dict:    Dictionary containing dataframe representations
                        of initial files
        
        - file:         Desired dataframe to visualize

    Returns:
        - Print statement of info
        - Print statement of statistics
        - Print statement of null values per column 
        - Print statement of duplicate rows
    """
    df = data_dict[file]

    # Info
    print(df.info())

    # Null per column
    null_per_column(df)

    # Duplicate rows 
    print(f"{file} # of duplicate rows: {df.duplicated().sum()}")

    # Statistics
    print(df.describe())

# I picture sorting the available dataframes into their respective dtype options then passing the array of columns in each datatype option into this function 
## These next couple functions should be in a function. Just need to figure out the Gradio formatting for user prompting. 
def handle_column_dtype(df, dtype_columns, dtype):
    for column in dtype_columns:
        df[column].astype(dtype)

    return df

# Updated null per column
# null_per_column(df)

# I picture sorting columns into drop or not drop bins.
def drop_columns(df, drop_columns):
    df = df.drop(drop_columns)

    return df

# Don't need a function for drop na
# df.dropna()

# Handle nulls: Actions include, mean, median, mode, random
def handle_nulls(df, action_columns, action):
    
    for column in action_columns:
        if action == 'Mean':
            df[column] = df[column].fillna(df[column].mean())
        elif action == 'Median':
            df[column] = df[column].fillna(df[column].median())
        elif action == 'Mode':
            df[column] = df[column].fillna(df[column].mode())
        else:
            if df[column].dtype == int:
                df[column] = df[column].fillna(random.randint(df[column].min, df[column].max))
            elif df[column].dtype == float:
                df[column] = df[column].fillna(random.uniform(df[column].min, df[column].max))
            else:
                df[column] = df[column].fillna(random.choice(df[column].unique()))

# Don't need a function to drop duplicates 
# df.drop_duplicates()

# Give summary of changes made and profiling_summary. Ask user if they are okay with this. If yes, add new cleaned dataframe to cleaned dataframe dictionary