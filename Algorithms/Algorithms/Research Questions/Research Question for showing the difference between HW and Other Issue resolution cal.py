import os
import pandas as pd
import numpy as np
from datetime import datetime
import random

# Your file paths
csv_file_folder1 = r'F:\Research\Final Help Wanted Research Datset and and Code\Another Analysis Dataset For RQ'
csv_file_folder = r'F:\RQ Modification\Pre-Solving Likelihood Dataset\Machine Learning Dataset Preparation For All Features Final'

def calculate_statistics(column_array):
    column_array = np.array(column_array, dtype=float)
    return {
        'min': np.min(column_array),
        'max': np.max(column_array),
        'median': np.median(column_array),
        'mean': np.mean(column_array)
    }

# Initialize lists for collecting data
final_help_wanted_data = []
final_other_issues_data = []
Url_commit_count = {} 
for subdir in os.listdir(csv_file_folder):
    print(subdir)
    file_path = os.path.join(csv_file_folder, subdir)
    
    df = pd.read_csv(file_path, low_memory=False)
    
    for index, row in df.iterrows():
        url = row['url']
        if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']):
            if str(row['state']) == 'closed' and str(row['issue_solver_commit_count']) != 'nan':
                Url_commit_count[row['url']] = True
            
# Process each CSV file in the folders
for subdir in os.listdir(csv_file_folder1):
    file_path = os.path.join(csv_file_folder1, subdir)
    df = pd.read_csv(file_path, low_memory=False)
    
    help_wanted_data = []
    other_issues_data = []
    for index, row in df.iterrows():
        string_to_label = str(row['label']).lower()
        if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']):
           if row['url'] in Url_commit_count and str(row['commitExist']) == 'False' and str(row['state']) == 'closed':
               help_wanted_data.append({
                   'resolution_time': row['resolution_time_after_the_tag_came']
               })
           elif str(row['commitExist']) == 'True' and str(row['state']) == 'closed':
               help_wanted_data.append({
                   'resolution_time': row['resolution_time_after_the_tag_came']
               })
        else:
            if row['url'] in Url_commit_count and str(row['commitExist']) == 'False' and str(row['state']) == 'closed':
                other_issues_data.append({
                    'resolution_time': row['resolution_time_after_the_tag_came']
                })
            elif str(row['commitExist']) == 'True' and str(row['state']) == 'closed':
                other_issues_data.append({
                    'resolution_time': row['resolution_time_after_the_tag_came']
                })


    # Randomly select 100 "other" issues for each "help wanted" issue and calculate means
    for hw_issue in help_wanted_data:
        if min(10, len(other_issues_data)) == 0:
               continue
        # Save the original help_wanted_issue in final_help_wanted
        final_help_wanted_data.append(hw_issue)
        
        # Randomly select a maximum of 100 'Other' issues (if available)
        random_other_issues = random.sample(other_issues_data, min(10, len(other_issues_data)))
        
        # Calculate the mean for the selected 'Other' issues
        means = {key: np.median([other[key] for other in random_other_issues]) for key in hw_issue}
        
        # Save the means in final_other_issue
        final_other_issues_data.append(means)
        print(len(final_help_wanted_data), len(final_other_issues_data), min(10, len(other_issues_data)))

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from math import sqrt

# Convert lists of dictionaries to DataFrames
help_wanted_df = pd.DataFrame(final_help_wanted_data)
other_issue_df = pd.DataFrame(final_other_issues_data)

# Function to calculate min, median, mean, max, p-value, and Cohen's d
def calculate_statistics(df1, df2):
    stats = []
    for column in df1.columns:
        # Calculate basic statistics
        min1, median1, mean1, max1 = df1[column].min(), df1[column].median(), df1[column].mean(), df1[column].max()
        min2, median2, mean2, max2 = df2[column].min(), df2[column].median(), df2[column].mean(), df2[column].max()
        
        # Perform independent t-test
        t_stat, p_value = ttest_ind(df1[column], df2[column], equal_var=False, nan_policy='omit')
        
        # Calculate Cohen's d
        mean_diff = mean1 - mean2
        pooled_std = sqrt(((df1[column].std() ** 2) + (df2[column].std() ** 2)) / 2)
        cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.nan

        # Append results
        stats.append({
            'Feature': column,
            'Help_Wanted_Min': min1,
            'Help_Wanted_Median': median1,
            'Help_Wanted_Mean': mean1,
            'Help_Wanted_Max': max1,
            'Other_Issue_Min': min2,
            'Other_Issue_Median': median2,
            'Other_Issue_Mean': mean2,
            'Other_Issue_Max': max2,
            'P_Value': p_value,
            'Cohen_D': cohen_d
        })
        
    return pd.DataFrame(stats)

# Calculate statistics
stats_df = calculate_statistics(help_wanted_df, other_issue_df)

# Save to CSV
output_path = "F:\\Research\\Results\\feature_statistics_resolution.csv"
stats_df.to_csv(output_path, index=False)

print(f"Statistics saved to {output_path}")
