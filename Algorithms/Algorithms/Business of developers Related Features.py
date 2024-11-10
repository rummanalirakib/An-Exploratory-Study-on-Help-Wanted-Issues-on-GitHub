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
import ast


def dateDifference(date1, date2):
    date1 = datetime.strptime(date1, "%Y-%m-%dT%H:%M:%SZ").date()
    date2 = datetime.strptime(date2, "%Y-%m-%dT%H:%M:%SZ").date()
    date_difference = date1 - date2
    return abs(date_difference.days)

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

def expertDeveloperGet(issueCreatedDate, commits):
    target_date = datetime.strptime(issueCreatedDate, '%Y-%m-%dT%H:%M:%SZ')
    filtered_commits = filter_commits_by_date(commits, target_date)
    top_committers = get_top_committers(filtered_commits)
    return top_committers

def count_dates_less_than(new_date_str, date_list):
    new_date = datetime.strptime(new_date_str, "%Y-%m-%dT%H:%M:%SZ")
    
    date_list_dt = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ") for date in date_list]
    
    count = sum(date <= new_date for date in date_list_dt)
    
    return count

def count_dates_in_range(dates, start_date, end_date):
    count = 0
    for date_str in dates:
        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        if start_date <= date <= end_date:
            count += 1
    return count

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
saved_folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Business of developers Calculation'

# Loop through each file in the folder
for subdir in os.listdir(root_directory):
    print(subdir)
    filename = subdir + '.csv'
    file_path = os.path.join(csv_file_folder, filename)
    file_path1 = os.path.join(root_directory, subdir)
    indexing = indexing + 1
    # Read the CSV file and specify the data types for columns
    dtypes = {'labels.name': str}  # Add other columns if necessary    
    
    path_exist = os.path.join(saved_folder_path, filename)
    if os.path.exists(path_exist):
        continue
    
    df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
    
    # Specify the value you want to check for in the specified column
    savedFile = os.path.join(saved_folder_path, filename)  # Corrected file path
    df1 = pd.DataFrame(columns=[
        'url', 'label', 'openedIRinHW', 'openedIRinOther', 
        'openedPR', 'numberofexpertdevelopers', 'howmanyexpertdeveloperscommittedrecently', 'numberofdevelopersassignedtoissues', 
        'numberofassignedissuestoexpertdevelopers', 'expertdevelopersrecentcommitcount', 'recentPRresolvingaveragetime', 'recentIRresolvingaveragetime'
    ])
        
    commentator = {}
    commentDate = {}
    
    comments_folder_path = os.path.join(file_path1, 'issues_comments')
    commits_folder_path = os.path.join(file_path1, 'commits')
    print(filename)            
    
    commits = {}
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

    
    Bugs = {}
    indx = 0
    count1 = count2 = 0
    OpenedHW = 0
    ClosedHw = []
    OpenedOther = 0
    ClosedOther = []
    OpenedPR = 0
    ClosedPR = []
    PRresolutionTime = []
    IRresolutionTime = []
    assignedIssueCount = []
    for index, row in df.iloc[::-1].iterrows():
        url = row['url']
        labels_name = str(row['labels.name'])
        
        if 'pull' in str(row['pull_request.html_url']):
            OpenedPR = OpenedPR + 1
            if str(row['closed_at']) != 'nan':
                ClosedPR.append(row['closed_at'])
                PRresolutionTime.append((dateDifference(row['created_at'], row['closed_at']), row['closed_at']))
                
        elif 'help' in str(row['labels.name']).lower() and 'wanted' in str(row['labels.name']).lower():
            OpenedHW = OpenedHW + 1
            if str(row['closed_at']) != 'nan':
                ClosedHw.append(row['closed_at'])
                
        else:
            OpenedOther = OpenedOther + 1
            if str(row['closed_at']) != 'nan':
                ClosedOther.append(row['closed_at'])
                IRresolutionTime.append((dateDifference(row['created_at'], row['closed_at']), row['closed_at']))
                
        if str(row['assignees.login']) != 'nan':
            if str(row['closed_at']) != 'nan':
                assignedIssueCount.append((row['assignees.login'], row['created_at'], row['closed_at']))
            else:
                assignedIssueCount.append((row['assignees.login'], row['created_at'], datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")))
            
        if labels_name == '' or labels_name == 'nan':
            continue
        
        if 'pull' in str(row['pull_request.html_url']):
            continue
        print(url)
        
        openedIRinHW = OpenedHW - count_dates_less_than(str(row['created_at']), ClosedHw)
        openedIRinOther = OpenedOther - count_dates_less_than(str(row['created_at']), ClosedOther)
        openedPR = OpenedPR - count_dates_less_than(str(row['created_at']), ClosedPR)
        
        expertDevelopers = expertDeveloperGet(row['created_at'], commits)
        
        current_date_str = row['created_at']
        current_date = datetime.strptime(current_date_str, "%Y-%m-%dT%H:%M:%SZ")

        past_30_days_start = current_date - timedelta(days=30)
        
        expert_developer_recent_commit_count = 0
        howmanyexpertdeveloperscommittedrecently = 0
        
        expert_developers = {}
        for committer, expert_value in expertDevelopers.items():
            dates = commits.get(committer, [])
            expert_developers[committer] = True
            count_in_range = count_dates_in_range(dates, past_30_days_start, current_date)
            expert_developer_recent_commit_count += count_in_range
            if count_in_range > 0:
                howmanyexpertdeveloperscommittedrecently += 1
            
        recentPRresolvingaveragetime = 0
        numberofPR = 0
        for value, date in PRresolutionTime:
            date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
            if past_30_days_start <= date <= current_date:
                recentPRresolvingaveragetime += value
                numberofPR += 1
                
        numberofPR = max(numberofPR, 1)
        recentPRresolvingaveragetime = recentPRresolvingaveragetime / numberofPR
        
        recentIRresolvingaveragetime = 0
        numberofIssue = 0
        for value, date in IRresolutionTime:
            date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
            if past_30_days_start <= date <= current_date:
                recentIRresolvingaveragetime += value
                numberofIssue += 1
                
        numberofIssue = max(numberofIssue, 1)
        recentIRresolvingaveragetime = recentIRresolvingaveragetime / numberofIssue
        
        
        numberofdevelopersassignedtoissues = 0
        numberofassignedissuestoexpertdevelopers = 0
        flagExpert = {}
        for value, created_date, closed_date in assignedIssueCount:
            created_date = datetime.strptime(created_date, "%Y-%m-%dT%H:%M:%SZ")
            closed_date = datetime.strptime(closed_date, "%Y-%m-%dT%H:%M:%SZ")
            if created_date > current_date:
                continue
            if closed_date == '':
                data_list = ast.literal_eval(value)
                for name in data_list:
                    if name in expert_developers:
                        numberofassignedissuestoexpertdevelopers += 1
                        if name not in flagExpert:
                            flagExpert[name] = True
                            numberofdevelopersassignedtoissues += 1
            elif current_date <= closed_date:
                data_list = ast.literal_eval(value)
                for name in data_list:
                    if name in expert_developers:
                        numberofassignedissuestoexpertdevelopers += 1
                        if name not in flagExpert:
                            flagExpert[name] = True
                            numberofdevelopersassignedtoissues += 1
        

        df1.loc[indx] = [
            row['url'],
            row['labels.name'],
            openedIRinHW,
            openedIRinOther,
            openedPR,
            len(expertDevelopers),
            expert_developer_recent_commit_count,
            numberofdevelopersassignedtoissues,
            numberofassignedissuestoexpertdevelopers,
            howmanyexpertdeveloperscommittedrecently,
            recentPRresolvingaveragetime,
            recentIRresolvingaveragetime
        ]
        indx = indx + 1
            
    df1.to_csv(savedFile, index=False, escapechar='\\')  
