import numpy as np
import re
import json
import requests
import os
import time
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statistics
import math


# Folder path containing the CSV files
root_directory = r'D:\Final Help Wanted Research Datset and and Code\HW Issues in csv file'

# List to store modified DataFrames
modified_dfs = []

new_folder_path = r'D:\Final Help Wanted Research Datset and and Code\HW Issues in csv file'
issueSolverIssueMean = []
issueSolverCommitMean = []
count1 = count2 = count3 = count4 = 0
hw_issues = {}
other_issues = {}
# Loop through each file in the folder
for subdir in os.listdir(new_folder_path):
    file_path1 = os.path.join(new_folder_path, subdir)
   # print('csv File: ', file_path1)
    try:
        df = pd.read_csv(file_path1, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        continue
    print(subdir)
    
    flag = 0
    count1 = 0 
    for index, row in df.iterrows():
        date_string = str(row['created_at']).strip()
        try:
            if date_string != 'nan' and date_string != '0':  # Checking for 'nan' and '0'
                dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
                year = dt.year
                string_to_label = str(row['labels.name']).lower()
                if 'help' in string_to_label and 'wanted' in string_to_label:
                    count1 += 1
            else:
                # Handle case where date_string is 'nan' or '0'
                # For example:
                year = None  # or any default value or handling you prefer
        except ValueError as e:
            # Handle ValueError (invalid isoformat string) if necessary
            print(f"Error parsing date: {e}")
            year = None  # or handle it accordingly

    hw_issues[subdir] = count1


sorted_hw_issues = dict(sorted(hw_issues.items(), key=lambda item: item[1], reverse=True))

# Extract keys and values from the sorted dictionary
repos = list(sorted_hw_issues.keys())
counts = list(sorted_hw_issues.values())

# Create short names for the repositories (first 3 characters)
short_names = [repo[:5] for repo in repos]

# Generate positions with gaps
x_positions = np.arange(len(short_names))  # Sequential positions for each bar
# Plotting the bar chart
plt.figure(figsize=(18, 8))
plt.bar(x_positions, counts, color='gray', width=0.5)  # Adjust width to create gaps
plt.xlabel('Repositories')
plt.ylabel('Help Wanted Issues Count')
plt.title('Help Wanted Issues Count per Repository')
plt.xticks(x_positions, short_names, rotation=45, ha='right')
plt.ylim(bottom=0)  # Start y-axis from 0
plt.xlim(-0.5, len(short_names) - 0.5)  # Adjust x-axis to include all categories

plt.tight_layout()

# Save the plot with high resolution
plt.savefig('help_wanted_issues_bar_chart.png', dpi=1200)

plt.show()