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

def count_urls_and_code_snippets(text):
    # Regular expression for detecting URLs
    url_pattern = r'https?://[^\s]+'
    # Regular expression for detecting code snippets (assuming they are enclosed in backticks or triple backticks)
    code_snippet_pattern = r'```[\s\S]*?```|`[^`]*`'

    # Find all matches for URLs and code snippets
    urls = re.findall(url_pattern, text)
    code_snippets = re.findall(code_snippet_pattern, text)

    # Return the counts of URLs and code snippets
    return len(urls), len(code_snippets)

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

file_path = '/home/rumman/Downloads/help_wanted_count_information_repositorywise_issues_comments.csv'

# Read the CSV file
df = pd.read_csv(file_path)
repo_name = {}
for index, row in df.iterrows():
    name = str(str(row['Name With Owner']).replace('/', '_')).lower()
    repo_name[name] = row['Stars Count']
    
# Loop through each file in the folder
root_directory = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Data'
csv_file_folder = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/HW Issues in csv file'
saved_folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Another Analysis Dataset For RQ'
count3=0
count4=0
# Loop through each file in the folder
for subdir in os.listdir(root_directory):
    print(subdir)
    filename = subdir + '.csv'
    file_path = os.path.join(csv_file_folder, filename)
    file_path1 = os.path.join(root_directory, subdir)
    
    df = pd.read_csv(file_path, low_memory=False)
    # Specify the value you want to check for in the specified column
    savedFile = os.path.join(saved_folder_path, filename)  # Corrected file path
    df1 = pd.DataFrame(columns=[
        'url', 'label', 'state', 'stars_count','commitExist', 'issue_label_changed_date', 'comments_list', 'resolution_time_after_the_tag_came'
    ])
        
    issue_events_folder_path = os.path.join(file_path1, 'issues_events')
    help_Wanted_date = {}
    Other_Label_Related_Date = {}
    issue_created_date = {}
    Commits_placed = {}
    
    commentator = {}
    
    comments_folder_path = os.path.join(file_path1, 'issues_comments')
    print(filename)            
    if os.path.exists(comments_folder_path) and os.path.isdir(comments_folder_path):
        for file_name in os.listdir(comments_folder_path):
            file_path = os.path.join(comments_folder_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as json_file:
                        item = json.load(json_file)
                        if isinstance(item, list):
                            for sub_item in item:
                                issue_url = sub_item.get('issue_url')
                                created_at = sub_item.get('created_at')
                                user_login = sub_item.get('user').get('login')
                                
                                if issue_url in commentator:
                                    commentator[issue_url] = commentator[issue_url] + "<" + user_login + ":" + created_at + ">"
                                else:
                                    commentator[issue_url] = "<" + user_login + ":" + created_at + ">"
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    else:
        print('issue comments Path Does not Exist')
    
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
                                    if event == 'labeled':
                                        if 'issue' in sub_item:
                                            url = sub_item.get('issue').get('url')
                                            labels = sub_item['issue'].get('labels', [])
                                            help_wanted_exists = False
                                            other_label_exists = False
                                            for label in labels:
                                                if str(label.get('name')).lower() == 'help' and str(label.get('name')).lower() == 'wanted':
                                                    help_wanted_exists = True
                                                elif str(label.get('name')).lower() == 'bug' or str(label.get('name')).lower() == 'Bug' or str(label.get('name')).lower() == 'enhancement':
                                                    other_label_exists = True
                                                
                                            if help_wanted_exists:
                                             #   print(url, sub_item.get('created_at'))
                                                help_Wanted_date[url] = sub_item.get('created_at')
                                            elif other_label_exists:
                                                if url not in Other_Label_Related_Date:
                                                    Other_Label_Related_Date[url] = sub_item.get('created_at')
                                                
                                            if sub_item.get('commit_id') is not None:
                                                Commits_placed[url] = True
                                        else:
                                            url = sub_item.get('url')
                                            help_wanted_exists = False
                                            other_label_exists = False
                                                
                                            if 'help' in str(sub_item.get('label').get('name')).lower() and 'wanted' in str(sub_item.get('label').get('name')).lower():
                                                help_Wanted_date[url] = sub_item.get('created_at')
                                            elif str(sub_item.get('label').get('name')).lower() == 'bug' or str(sub_item.get('label').get('name')).lower() == 'Bug' or str(sub_item.get('label').get('name')).lower() == 'enhancement':
                                                if url not in Other_Label_Related_Date:
                                                    Other_Label_Related_Date[url] = sub_item.get('created_at')
                                                
                                            if sub_item.get('commit_id') is not None:
                                                    Commits_placed[url] = True
                                                                                    
                                    else:
                                        if 'issue' in sub_item:
                                            url = sub_item.get('issue').get('url')

                                            if sub_item.get('commit_id') is not None:
                                                Commits_placed[url] = True
                                        else:
                                            url = sub_item.get('url')
                                            
                                            if sub_item.get('commit_id') is not None:
                                                Commits_placed[url] = True
                                                
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    
    wantedBug = {}
    unWantedBug = {}
    Bugs = {}
    indx = 0
    count1 = count2 = 0
    for index, row in df.iterrows():
        url = row['url']
        print(url)
        if str(row['labels.name']) == 'nan':
            continue
            
        commitExist = False
        issue_label_changed_date = ''
        comments = ''
        if url in Commits_placed:
            commitExist = True
            
        if url in help_Wanted_date:
            issue_label_changed_date = help_Wanted_date[url] 
        elif url in Other_Label_Related_Date:
            issue_label_changed_date = Other_Label_Related_Date[url] 
        
        if url in commentator:
            comments = commentator[url]
            
        resolution_time = ''
        
        if str(row['closed_at']) != 'nan':
            if url in help_Wanted_date:
                resolution_time = dateDifference(row['closed_at'], help_Wanted_date[url])
            elif url in Other_Label_Related_Date:
                resolution_time = dateDifference(row['closed_at'], Other_Label_Related_Date[url])
        
        if issue_label_changed_date != '' and ('bug' in str(row['labels.name']).lower() or 'enhancement' in str(row['labels.name']).lower() or ('help' in str(row['labels.name']).lower() and 'wanted' in str(row['labels.name']).lower())):
            
            df1.loc[indx] = [
                row['url'],
                row['labels.name'],
                row['state'],
                repo_name[str(subdir).lower()],
                commitExist,
                issue_label_changed_date,
                comments,
                resolution_time
            ]
            indx = indx + 1
            
    print(count1, count2, count3, count4)
    df1.to_csv(savedFile, index=False, escapechar='\\')  
