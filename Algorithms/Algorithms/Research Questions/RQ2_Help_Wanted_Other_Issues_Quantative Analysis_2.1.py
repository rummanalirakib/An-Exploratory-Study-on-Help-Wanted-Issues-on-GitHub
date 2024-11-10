import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random
# Function to calculate statistics with specific conditions
def calculate_statistics(column_array):
    # Convert column_array to numerical type before calculating statistics
    column_array = np.array(column_array, dtype=float)
    return {
        'min': np.min(column_array),
        'max': np.max(column_array),
        'median': np.median(column_array),
        'mean': np.mean(column_array)
    }

# Folder path containing the CSV files
folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Dataset Final Preparation for Machine Learning'


commentatorNumber_hw = []
commentatorNumber_gfi = []

commentNumber_hw = []
commentNumber_gfi = []

resolution_time_hw = []
resolution_time_gfi = []


mentioned_developer_hw = []
mentioned_developer_gfi = []

subscribed_hw = []
subscribed_gfi = []

expert_comments_hw = []

mid_level_comments_hw = []

newcomer_comments_hw = []
count1 = count2 = 0
# Loop through each file in the folder
for filename in os.listdir(folder_path):
    
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
       # print(filename)
        # Load the CSV file
        dtypes = {'label': str}  
        df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)


        for index, row in df.iterrows():
            if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']).lower():
           # if not ('help' in str(row['label']).lower() and 'wanted' in str(row['label']).lower()):
                count1 = count1 + 1
                data_string = str(row['commentators'])
                # Split the string by '>'
                entries = data_string.split(">")

                # Extract names and store in a set to ensure uniqueness
                unique_names = set(entry.split(":")[0][1:] for entry in entries if entry)

                # Get the unique name count
                unique_name_count = len(unique_names)
                
                commentatorNumber_hw.append(unique_name_count)
                
                commentNumber_hw.append(row['commentNumber'])
                
                if row['resolution time'] != -1:
                    resolution_time_hw.append(row['resolution time'])
                    count2 += 1
                
                mentioned_developer_hw.append(row['mentioned_developer'])
                
                subscribed_hw.append(row['subscriber'])
                
                expert_comments_hw.append(row['expert_comments'])
                
                mid_level_comments_hw.append(row['mid_level_comments'])
                
                newcomer_comments_hw.append(row['newcomer_comments'])
                
               # print(count1)


    
# Calculate overall statistics from cumulative data
help_wanted_stats = {
    'commentatorNumber': calculate_statistics(commentatorNumber_hw),
    'commentNumber': calculate_statistics(commentNumber_hw),
    'resolution time': calculate_statistics(resolution_time_hw),
    'mentioned_developer': calculate_statistics(mentioned_developer_hw),
    'subscribed': calculate_statistics(subscribed_hw),
    'expert_comments_hw': calculate_statistics(expert_comments_hw),
    'mid_level_comments_hw': calculate_statistics(mid_level_comments_hw),
    'newcomer_comments_hw': calculate_statistics(newcomer_comments_hw)
}


# Print the results
print("Statistics for 'help wanted':")
for col, stats in help_wanted_stats.items():
    print(f"{col}: {stats}")