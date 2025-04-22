import os
import pandas as pd

# Folder path
folder_path = 'E:/WorkSpace/Research/ITSCPaper/NewCode'

# Read all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Load each CSV file into a DataFrame
dataframes = {}
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    dataframes[csv_file] = pd.read_csv(file_path)

# Print the names of the loaded files
print(f"Loaded CSV files: {list(dataframes.keys())}")