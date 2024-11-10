import os
import pandas as pd
import numpy as np
from datetime import datetime
import random

# Your file paths
csv_file_folder1 = r'F:\Research\Final Help Wanted Research Datset and and Code\Another Analysis Dataset For RQ'
csv_file_folder = r'F:\RQ Modification\Pre-Solving Likelihood Dataset\Machine Learning Dataset Preparation For All Features Final'

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

# Process each CSV file in the folders
for subdir in os.listdir(csv_file_folder1):
    file_path = os.path.join(csv_file_folder1, subdir)
    df = pd.read_csv(file_path, low_memory=False)
    
    help_wanted_data = []
    other_issues_data = []
    for index, row in df.iterrows():
        string_to_label = str(row['label']).lower()
        if 'help' in string_to_label and 'wanted' in string_to_label:

            count_after_1_day, count_after_3_days, count_after_7_days, count_before, count_after_all_days, unique_1_day, unique_3_days, unique_7_days, unique_commenters_all = count_dates_after_and_before_given(str(row['comments_list']), row['issue_label_changed_date'])

            help_wanted_data.append({
               # 'resolution_time': row['resolution_time_after_the_tag_came'],
                'comments_after': count_after_all_days,
                'comments_before': count_before,
                'comment_after_1_day': count_after_1_day,
                'comment_after_3_days': count_after_3_days,
                'comment_after_7_days': count_after_7_days,
                'unique_commenters_1_day': unique_1_day,
                'unique_commenters_3_days': unique_3_days,
                'unique_commenters_7_days': unique_7_days,
                'unique_commenters_all': unique_commenters_all
            })
        else:
            count_after_1_day, count_after_3_days, count_after_7_days, count_before, count_after_all_days, unique_1_day, unique_3_days, unique_7_days, unique_commenters_all = count_dates_after_and_before_given(str(row['comments_list']), row['issue_label_changed_date'])

            other_issues_data.append({
               # 'resolution_time': row['resolution_time_after_the_tag_came'],
                'comments_after': count_after_all_days,
                'comments_before': count_before,
                'comment_after_1_day': count_after_1_day,
                'comment_after_3_days': count_after_3_days,
                'comment_after_7_days': count_after_7_days,
                'unique_commenters_1_day': unique_1_day,
                'unique_commenters_3_days': unique_3_days,
                'unique_commenters_7_days': unique_7_days,
                'unique_commenters_all': unique_commenters_all
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
output_path = "F:\\Research\\Results\\feature_statistics_traffic.csv"
stats_df.to_csv(output_path, index=False)

print(f"Statistics saved to {output_path}")
