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

def dateDifference(date1, date2):
    # Input date strings
    date_str1 = date1
    date_str2 = date2
    
    if date_str1 == 'nan' or date_str2 == 'nan':
        return 120
    
    # Parse the date strings into date objects
    date1 = datetime.strptime(date_str1, "%Y-%m-%dT%H:%M:%SZ").date()
    date2 = datetime.strptime(date_str2, "%Y-%m-%dT%H:%M:%SZ").date()
    
    # Calculate the date difference
    date_difference = date1 - date2
    return abs(date_difference.days)

def get_recent_bug_count(df, start_index, key):
    rBN = 0
    for next_index, next_row in df.iloc[start_index + 1:].iterrows():
        temp = dateDifference(str(df.loc[start_index, 'created_at']), str(next_row['created_at']))
        if temp > 90:
            break
        if key == str(next_row['user.login']):
            rBN = rBN + 1
    return rBN


# List to store modified DataFrames
modified_dfs = []
count3 = count4 = 0
title = []
body = []
label = []
keyword_counts = {}
Reporter_Experience = []
count5=0
indexing = 0
allIndex = 0

# Loop through each file in the folder
root_directory = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Data'
csv_file_folder = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/HW Issues in csv file'
new_folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Dataset Preparation for Machine Learning'
saved_folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Dataset Final Preparation for Machine Learning'


# Loop through each file in the folder
for subdir in os.listdir(root_directory):
    if subdir != '24pullrequests_24pullrequests':
        continue
    print(subdir)
    filename = subdir + '.csv'
    file_path = os.path.join(csv_file_folder, filename)
    file_path1 = os.path.join(root_directory, subdir)
    update_moreinfo_path = os.path.join(new_folder_path, filename)
    indexing = indexing + 1
    # Read the CSV file and specify the data types for columns
    dtypes = {'labels.name': str}  # Add other columns if necessary    
    
    df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
    # Specify the column name you want to check
    column_name_to_check = 'labels.name'
    title = {}
    body = {}
    reporter = {}
    created_at_date = {}
    for index, row in df.iloc[::-1].iterrows():
        title[row['url']] = row['title']
        body[row['url']] = row['body']
        reporter[row['url']] = row['user.login']
        created_at_date[row['url']] = row['created_at']
    
    # Specify the value you want to check for in the specified column
    savedFile = os.path.join(saved_folder_path, filename)  # Corrected file path
    df1 = pd.DataFrame(columns=[
        'url', 'title', 'body', 'label', 'reporter', 'prev30DaysComments', 'subscriber', 'lengthofthetitle', 'lengthofthedesc', 'reporterlevel', 'bugnum', 'latest_bugnum', 'hw_ratio', 'commentators', 'commentNumber', 'state', 'resolution time', 'mentioned_developer', 'assigned_issue_count', 'latest_assigned_issue_count', 'issue_solver', 'issue_solver_commit_count', 'issue_solver_level', 'expert_comments', 'mid_level_comments', 'newcomer_comments'
    ])

        
    issue_events_folder_path = os.path.join(file_path1, 'issues_events')
    print(issue_events_folder_path)
    Subscriber = {}
    if os.path.exists(issue_events_folder_path) and os.path.isdir(issue_events_folder_path):
        for file_name in os.listdir(issue_events_folder_path):
            file_path = os.path.join(issue_events_folder_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as json_file:
                        item = json.load(json_file)
                        if isinstance(item, list):
                            for sub_item in item:
                                if sub_item.get('event'):
                                    event = sub_item.get('event')
                                    if event == 'subscribed':
                                        url = sub_item.get('issue', {}).get('url')
                                        if not url:
                                            url = sub_item.get('url')
                                        if url in Subscriber:
                                            Subscriber[url] += 1
                                        else:
                                            Subscriber[url] = 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    
    wantedBug = {}
    unWantedBug = {}
    Bugs = {}
    indx = 0
    count1 = count2 = 0
    # Iterate over each row in the DataFrame
    print(update_moreinfo_path)
    df = pd.read_csv(update_moreinfo_path, low_memory=False)
    for index, row in df.iterrows():
        url = row['url']

        print(url)
        key = reporter[url]

        subscriber = 0
        if url in Subscriber:
            subscriber = Subscriber[url]
        
        if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']).lower():

            if key in wantedBug:
                wantedBug[key] = wantedBug[key] + 1
            else:
                wantedBug[key] = 1

            denominator = 1 if unWantedBug.get(key, 0) + (wantedBug.get(key, 0) - 1) == 0 else (unWantedBug.get(key, 0) + wantedBug.get(key, 0) - 1)
            df1.loc[indx] = [
                url, 
                title[url], 
                body[url], 
                row['label'], 
                key, 
                row['prev30DaysComments'], 
                subscriber, 
                row['lengthofthetitle'], 
                row['lengthofthedesc'], 
                row['reporterlevel'], 
                row['bugnum'], 
                row['latest_bugnum'], 
                (wantedBug[key] - 1) / denominator,
                row['commentators'], 
                row['commentNumber'], 
                row['state'], 
                row['resolution time'], 
                row['mentioned_developer'], 
                row['assigned_issue_count'], 
                row['latest_assigned_issue_count'], 
                row['issue_solver'], 
                row['issue_solver_commit_count'], 
                row['issue_solver_level'], 
                row['expert_comments'], 
                row['mid_level_comments'], 
                row['newcomer_comments']
            ]
            indx = indx + 1
            count1 = count1 + 1
        else:

            if key in unWantedBug:
                unWantedBug[key] = unWantedBug[key] + 1
            else:
                unWantedBug[key] = 1

            denominator = 1 if (unWantedBug.get(key, 0) - 1) + wantedBug.get(key, 0) == 0 else (unWantedBug.get(key, 0) - 1 + wantedBug.get(key, 0))
            numerator = 0 if wantedBug.get(key, 0) == 0 else wantedBug.get(key)
            df1.loc[indx] = [
                url, 
                title[url], 
                body[url], 
                row['label'], 
                key, 
                row['prev30DaysComments'], 
                subscriber, 
                row['lengthofthetitle'], 
                row['lengthofthedesc'], 
                row['reporterlevel'], 
                row['bugnum'], 
                row['latest_bugnum'], 
                numerator / denominator,
                row['commentators'], 
                row['commentNumber'], 
                row['state'], 
                row['resolution time'], 
                row['mentioned_developer'], 
                row['assigned_issue_count'], 
                row['latest_assigned_issue_count'], 
                row['issue_solver'], 
                row['issue_solver_commit_count'], 
                row['issue_solver_level'], 
                row['expert_comments'], 
                row['mid_level_comments'], 
                row['newcomer_comments']
            ]
            indx = indx + 1
        
    df1.to_csv(savedFile, index=False, escapechar='\\')  
