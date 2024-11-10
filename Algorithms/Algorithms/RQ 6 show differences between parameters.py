import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from sklearn.model_selection import KFold
import json
from datetime import datetime, timedelta
import re
import numpy as np
import random
from statistics import mean, stdev
from math import sqrt

# List to store modified DataFrames
modified_dfs = []
count3 = count4 = 0
title = []
body = []
label = []
keyword_counts = {}
Reporter_Experience = []
count5 = 0
indexing = 0
allIndex = 0

file_path = '/home/rumman/Downloads/help_wanted_count_information_repositorywise_issues_comments.csv'

def count_dates_after_and_before_given(comment, given_date):
    # Initialize counters for different intervals
    count_after_1_day = 0
    count_after_3_days = 0
    count_after_7_days = 0
    count_after_all_days = 0
    count_before = 0
    
    # Track unique commenters within each interval
    unique_commenters_1_day = set()
    unique_commenters_3_days = set()
    unique_commenters_7_days = set()
    unique_commenters_all = set()
    
    given_date = datetime.strptime(str(given_date), "%Y-%m-%dT%H:%M:%SZ")
    
    # Split the comment into potential date parts using '>' as delimiter
    for part in comment.split('>'):
        parts = part.split(':', 1)
        
        if len(parts) > 1:
            date_value_str = parts[1]
            date_value = datetime.strptime(date_value_str, "%Y-%m-%dT%H:%M:%SZ")
            time_diff = date_value - given_date
            
            # Check intervals and update counters and unique commenter sets
            if time_diff.days < 0:
                count_before += 1
            else:
                commenter_id = parts[0]  # Assuming the commenter's ID or identifier is before the date
                if time_diff.days <= 7:
                    count_after_1_day += 1
                    unique_commenters_1_day.add(commenter_id)
                if time_diff.days <= 14:
                    count_after_3_days += 1
                    unique_commenters_3_days.add(commenter_id)
                if time_diff.days <= 21:
                    count_after_7_days += 1
                    unique_commenters_7_days.add(commenter_id)
                count_after_all_days += 1
                unique_commenters_all.add(commenter_id)
    
    return (count_after_1_day, count_after_3_days, count_after_7_days, 
            count_before, count_after_all_days, len(unique_commenters_1_day), 
            len(unique_commenters_3_days), len(unique_commenters_7_days), len(unique_commenters_all))

def select_random_samples(array, num_samples):
    # Convert array to a NumPy array if it's not already
    array = np.array(array)
    
    # Check if the number of samples is less than or equal to the size of the array
    if num_samples > len(array):
        raise ValueError("Number of samples exceeds the number of elements in the array")
    
    # Select random samples
    random_samples = np.random.choice(array, size=num_samples, replace=False)
    
    return random_samples

# Read the CSV file
df = pd.read_csv(file_path)
repo_name = {}
for index, row in df.iterrows():
    name = str(str(row['Name With Owner']).replace('/', '_')).lower()
    repo_name[name] = row['Stars Count']

# Initialize counters
count1 = count5 = count6 = 0
count2 = count3 = count4 = 0
Url_commit_count = {}

# Loop through each file in the folder
csv_file_folder1 = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Another Analysis Dataset For RQ'
csv_file_folder = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Machine Learning Dataset Preparation For All Features Final'

for subdir in os.listdir(csv_file_folder):
    print(subdir)
    file_path = os.path.join(csv_file_folder, subdir)
    
    df = pd.read_csv(file_path, low_memory=False)
    
    for index, row in df.iterrows():
        url = row['url']
        if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']):
            count1 += 1
            if str(row['state']) == 'closed':
                count4 += 1
            if str(row['state']) == 'closed' and str(row['issue_solver_commit_count']) != 'nan':
                count3 += 1
                Url_commit_count[row['url']] = True
        else:
            count2 += 1

print('NEW DATASET READ START:')
resolution_after_HW_tag_came = []
resolution_after_other_tag_came = []
Comments_after_HW_tag_came = []
Comments_after_Other_tag_came = []
Comments_before_HW_tag_came = []
Comments_before_Other_tag_came = []

# Initialize counters for comment intervals
comment_after_1_day_HW = []
comment_after_3_days_HW = []
comment_after_7_days_HW = []

comment_after_1_day_Other = []
comment_after_3_days_Other = []
comment_after_7_days_Other = []

unique_commenters_after_1_days_HW = []
unique_commenters_after_3_days_HW = []
unique_commenters_after_7_days_HW = []
unique_commenters_after_ALL_HW = []

unique_commenters_after_1_days_Other = []
unique_commenters_after_3_days_Other = []
unique_commenters_after_7_days_Other = []
unique_commenters_after_ALL_Other = []

for subdir in os.listdir(csv_file_folder1):
    print(subdir)
    file_path = os.path.join(csv_file_folder1, subdir)
    df = pd.read_csv(file_path, low_memory=False)
    
    for index, row in df.iterrows():    
        if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']):
            flag = 0
            if row['url'] in Url_commit_count and str(row['commitExist']) == 'False' and str(row['state']) == 'closed':
                count5 += 1
                resolution_after_HW_tag_came.append(row['resolution_time_after_the_tag_came'])
                flag = 1
            elif str(row['commitExist']) == 'True' and str(row['state']) == 'closed':
                count6 += 1
                resolution_after_HW_tag_came.append(row['resolution_time_after_the_tag_came'])
                flag = 1
            
            if flag == 1:
                count_after_1_day, count_after_3_days, count_after_7_days, count_before, count_after_all_days, unique_1_day, unique_3_days, unique_7_days, unique_commenters_all = count_dates_after_and_before_given(str(row['comments_list']), row['issue_label_changed_date'])
                Comments_after_HW_tag_came.append(count_after_all_days)
                Comments_before_HW_tag_came.append(count_before)
                comment_after_1_day_HW.append(count_after_1_day)
                comment_after_3_days_HW.append(count_after_3_days)
                comment_after_7_days_HW.append(count_after_7_days)
                unique_commenters_after_1_days_HW.append(unique_1_day)
                unique_commenters_after_3_days_HW.append(unique_3_days)
                unique_commenters_after_7_days_HW.append(unique_7_days)
                unique_commenters_after_ALL_HW.append(unique_commenters_all)

        else:
            flag = 0
            if str(row['commitExist']) == 'True' and str(row['state']) == 'closed':
                resolution_after_other_tag_came.append(row['resolution_time_after_the_tag_came'])
                flag = 1
            if flag == 1:
                count_after_1_day, count_after_3_days, count_after_7_days, count_before, count_after_all_days, unique_1_day, unique_3_days, unique_7_days, unique_commenters_all = count_dates_after_and_before_given(str(row['comments_list']), row['issue_label_changed_date'])
                Comments_after_Other_tag_came.append(count_after_all_days)
                Comments_before_Other_tag_came.append(count_before)
                comment_after_1_day_Other.append(count_after_1_day)
                comment_after_3_days_Other.append(count_after_3_days)
                comment_after_7_days_Other.append(count_after_7_days)
                unique_commenters_after_1_days_Other.append(unique_1_day)
                unique_commenters_after_3_days_Other.append(unique_3_days)
                unique_commenters_after_7_days_Other.append(unique_7_days)
                unique_commenters_after_ALL_Other.append(unique_commenters_all)




def calculate_statistics(column_array):
    # Convert column_array to numerical type before calculating statistics
    column_array = np.array(column_array, dtype=float)
    return {
       # 'min': np.min(column_array),
      #  'max': np.max(column_array),
        'median': np.median(column_array),
        'mean': np.mean(column_array)
    }

help_wanted_and_other_stats = {
    'resolution_after_HW_tag_came': calculate_statistics(resolution_after_HW_tag_came),
    'resolution_after_other_tag_came': calculate_statistics(resolution_after_other_tag_came),
    
    'Comments_after_HW_tag_came': calculate_statistics(Comments_after_HW_tag_came),
    'Comments_before_HW_tag_came': calculate_statistics(Comments_before_HW_tag_came),
    'comment_after_1_day_HW': calculate_statistics(comment_after_1_day_HW),
    'comment_after_3_days_HW': calculate_statistics(comment_after_3_days_HW),
    'comment_after_7_days_HW': calculate_statistics(comment_after_7_days_HW),
    'unique_commenters_after_1_days_HW': calculate_statistics(unique_commenters_after_1_days_HW),
    'unique_commenters_after_3_days_HW': calculate_statistics(unique_commenters_after_3_days_HW),
    'unique_commenters_after_7_days_HW': calculate_statistics(unique_commenters_after_7_days_HW),
    'unique_commenters_after_ALL_HW': calculate_statistics(unique_commenters_after_ALL_HW),
    
    'Comments_after_Other_tag_came': calculate_statistics(select_random_samples(Comments_after_Other_tag_came, len(Comments_after_HW_tag_came))),
    'Comments_before_Other_tag_came': calculate_statistics(Comments_before_Other_tag_came),
    'comment_after_1_day_Other': calculate_statistics(comment_after_1_day_Other),
    'comment_after_3_days_Other': calculate_statistics(comment_after_3_days_Other),
    'comment_after_7_days_Other': calculate_statistics(comment_after_7_days_Other),
    'unique_commenters_after_1_days_Other': calculate_statistics(unique_commenters_after_1_days_Other),
    'unique_commenters_after_3_days_Other': calculate_statistics(unique_commenters_after_3_days_Other),
    'unique_commenters_after_7_days_Other': calculate_statistics(unique_commenters_after_7_days_Other),
    'unique_commenters_after_ALL_Other': calculate_statistics(unique_commenters_after_ALL_Other)
    
}

print("Statistics for 'help wanted':")
for col, stats in help_wanted_and_other_stats.items():
    print(f"{col}: {stats}")


from scipy.stats import mannwhitneyu

def mann_whitney_u_test(group1, group2):
    # Convert groups to numpy arrays if they aren't already
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    # Perform Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Calculate effect size r
    n1 = len(group1)
    n2 = len(group2)
    r = abs(u_stat - (n1 * n2) / 2) / (n1 * n2)
    
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    stddev1 = np.std(group1, ddof=1)
    stddev2 = np.std(group2, ddof=1)
    
    pooled_stddev = np.sqrt((stddev1 ** 2 + stddev2 ** 2) / 2)
    
    # Cohen's d calculation (for effect size comparison)
    cohen_d = (mean1 - mean2) / pooled_stddev
    
    return u_stat, p_value, r, cohen_d

# Metrics for "help wanted" and "other" issues
metrics = [
    ('resolution_after_HW_tag_came', resolution_after_HW_tag_came, resolution_after_other_tag_came),
    ('Comments_after_HW_tag_came', Comments_after_HW_tag_came, Comments_after_Other_tag_came),
    ('Comments_before_HW_tag_came', Comments_before_HW_tag_came, Comments_before_Other_tag_came),
    ('comment_after_1_day_HW', comment_after_1_day_HW, comment_after_1_day_Other),
    ('comment_after_3_days_HW', comment_after_3_days_HW, comment_after_3_days_Other),
    ('comment_after_7_days_HW', comment_after_7_days_HW, comment_after_7_days_Other),
    ('unique_commenters_after_1_days_HW', unique_commenters_after_1_days_HW, unique_commenters_after_1_days_Other),
    ('unique_commenters_after_3_days_HW', unique_commenters_after_3_days_HW, unique_commenters_after_3_days_Other),
    ('unique_commenters_after_7_days_HW', unique_commenters_after_7_days_HW, unique_commenters_after_7_days_Other),
    ('unique_commenters_after_ALL_HW', unique_commenters_after_ALL_HW, unique_commenters_after_ALL_Other)
]

# Calculate and print results
print("Mann-Whitney U test and Effect Size for metrics:")
for metric_name, hw_values, other_values in metrics:
    u_stat, p_value, r, cohenD = mann_whitney_u_test(hw_values, other_values)
    print(f"{metric_name}:")
    print(f"  Mann-Whitney U statistic: {u_stat}")
    print(f"  P-value: {p_value}")
    print(f"  Effect size (r): {r}")
    print(f"cohen'd : {cohenD}")