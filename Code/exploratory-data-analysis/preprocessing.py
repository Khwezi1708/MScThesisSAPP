import pandas as pd
# Here we will combine the datasets and create new features where necessary. A
# Dataset location
file_pattern_ZRA = '../../Data/ZRA'

# Read the Excel file
nana = pd.read_csv(f'{file_pattern_ZRA}/Nanas Farm Daily Flows 1994 to 2025.csv',delimiter=';')
nana


