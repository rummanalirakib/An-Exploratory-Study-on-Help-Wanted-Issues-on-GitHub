import os
import pandas as pd
import re
from datetime import datetime, timedelta
import json
from nltk.tokenize import word_tokenize
import requests
import time
        
def process_and_write_comments(issue_comments, issue_url):
    commentatorNumber = {}

    if contains_help_wanted(issue_comments):
        return 1
            
    return 0
      

def count_dates_within_range(assignee, target_date, date_dict, range_days):
    if assignee not in date_dict:
        return 0

    # Convert the target date to a datetime object
    target_date = datetime.strptime(target_date, '%Y-%m-%dT%H:%M:%SZ')

    # Calculate the date range
    start_date = target_date - timedelta(days=range_days)
    end_date = target_date

    # Parse the dates for the specific assignee
    assignee_date_list = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ') for date in date_dict[assignee]]
    # Count the dates within the range
    count = sum(start_date <= date < end_date for date in assignee_date_list)

    return count

def filter_commits_by_date(commits, target_date):
    # Calculate the start and end dates for the previous calendar year relative to target_date
    previous_year_start = datetime(target_date.year - 1, 1, 1)
    previous_year_end = datetime(target_date.year - 1, 12, 31, 23, 59, 59)
    
    filtered_commits = {}
    
    for committer, dates in commits.items():
        filtered_dates = [
            date for date in dates 
            if previous_year_start <= datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ') <= previous_year_end
        ]
        if filtered_dates:
            filtered_commits[committer] = filtered_dates
    
    return filtered_commits

def get_top_committers(filtered_commits, top_percentage=20):
    commit_counts = {committer: len(dates) for committer, dates in filtered_commits.items()}
    sorted_committers = sorted(commit_counts.items(), key=lambda item: item[1], reverse=True)
    top_n = int(len(sorted_committers) * (top_percentage / 100))
    top_committers = sorted_committers[:top_n]
    return dict(top_committers)

def categorize_committer(committer_name, top_committers, filtered_commits, issueSolverCommitCount):
    if issueSolverCommitCount < 3:
        return 'newcomer'
    elif committer_name in top_committers:
       return "expert"
    else:
       return "mid"


def committerLevelDecide(issueCreatedDate, commits, issueSolverName, issueSolverCommitCount):
    target_date = datetime.strptime(issueCreatedDate, '%Y-%m-%dT%H:%M:%SZ')
    filtered_commits = filter_commits_by_date(commits, target_date)
    top_committers = get_top_committers(filtered_commits)
    result = categorize_committer(issueSolverName, top_committers, filtered_commits, issueSolverCommitCount)
    return result

def extract_issue_number(url):
    match = re.search(r'issues/(\d+)', url)
    if match:
        return match.group(1)
    else:
        return None

def contains_help_wanted(text):
    # Convert the text to lowercase
    text_lower = text.lower()
    
    # Regular expression to match 'help' followed by zero or more spaces and then 'wanted'
    pattern = re.compile(r'help\s*wanted')
    
    # Search for the pattern in the text
    return bool(pattern.search(text_lower)) 


# Folder path containing the CSV files
root_directory = r'D:\Final Help Wanted Research Datset and and Code\Data'
count2 = count3 = count4 = 0
final_github_access_token = ""


df1 = pd.DataFrame(columns=[
    'url', 'CommentatorExperience'
])
savedFile = os.path.join('D:\Final Help Wanted Research Datset and and Code\HW Tag Found In Issues and Body', 'HWI_HAVING_HW_TAG_WITH_COMMENTATOR.csv')
indx = 0
# Loop through each file in the folder
for subdir in os.listdir(root_directory):
    subdir_path = os.path.join(root_directory, subdir)
    commits_folder_path = os.path.join(subdir_path, 'commits')
    comments_folder_path = os.path.join(subdir_path, 'issues_comments')
    issue_events_path = os.path.join(subdir_path, 'issues_events')
    filename = str(subdir).lower() + '.csv'
    print(filename)            
    commits = {}
    comments = {}
    if os.path.exists(commits_folder_path) and os.path.isdir(commits_folder_path):
        for file_name in os.listdir(commits_folder_path):
            file_path = os.path.join(commits_folder_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as json_file:
                        item = json.load(json_file)
                        if isinstance(item, list):
                            for sub_item in item:
                                if sub_item.get('author'):
                                    committer = sub_item['author'].get('login')
                                    committer_date = sub_item['commit']['author'].get('date')
                                   # print('Committer: ', committer, committer_date)
                                    if committer in commits:
                                        commits[committer].append(committer_date)
                                    else:
                                        commits[committer] = [committer_date]
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    else:
        print('Commit Path Does not Exist')
        
    
    if os.path.exists(comments_folder_path) and os.path.isdir(comments_folder_path):
        for file_name in os.listdir(comments_folder_path):
            file_path = os.path.join(comments_folder_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as json_file:
                        item = json.load(json_file)
                        if isinstance(item, list):
                            for sub_item in item:
                                if sub_item['issue_url'] in comments:
                                   comments[sub_item['issue_url']] += sub_item['body']
                                else:
                                   comments[sub_item['issue_url']] = sub_item['body']

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    else:
        print('Comments Path Does not Exist')
    
    modified_subdir = subdir.replace('_', '/')
    file_path = os.path.join('D:\Final Help Wanted Research Datset and and Code\HW Issues in csv file', filename)
    
    if not os.path.exists(file_path):
        continue
    # Load the CSV file
    dtypes = {'url': str}  
    df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
    for index, row in df.iterrows():
        string_to_label = str(row['labels.name']).lower()
        if 'help' in string_to_label and 'wanted' in string_to_label:
            flag1 = contains_help_wanted(str(row['body']))
            issue_comments = comments.get(row['url'], "")
                        
            flag = process_and_write_comments(issue_comments, row['url'])            
            if flag == 1 or flag1 == 1:
                with open(savedFile, 'a') as f:
                    f.write(f'{row["url"]}\n')
