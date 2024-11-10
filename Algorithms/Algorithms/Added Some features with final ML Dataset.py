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
new_folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Machine Learning Dataset For All Features'
saved_folder_path = r'//home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Machine Learning Dataset Preparation For All Features Final'
count3=0
count4=0
# Loop through each file in the folder
for subdir in os.listdir(root_directory):
    print(subdir)
    filename = subdir + '.csv'
    file_path = os.path.join(csv_file_folder, filename)
    file_path1 = os.path.join(root_directory, subdir)
    update_moreinfo_path = os.path.join(new_folder_path, filename)
    
    df = pd.read_csv(file_path, low_memory=False)
    # Specify the column name you want to check
    column_name_to_check = 'labels.name'
    created_at_date = {}
    for index, row in df.iterrows():
        created_at_date[row['url']] = row['created_at']
    
    # Specify the value you want to check for in the specified column
    savedFile = os.path.join(saved_folder_path, filename)  # Corrected file path
    df1 = pd.DataFrame(columns=[
        'url', 'title', 'body', 'label', 'number_of_codesnippet', 'number_of_urls', 
        'HW_tag_came_after_how_many_days', 'HW_resolving_time_after_the_tag_came',
        'prev30DaysComments', 'bugnum', 'latest_bugnum', 'hw_ratio', 'reporter_experience_level', 
        'assigned_issue_count', 'latest_assigned_issue_count', 'state', 'mentioned_developer', 
        'subscriber', 'lengthofthetitle', 'lengthofthedesc', 'commentNumber', 'resolution time',
        'issue_solver', 'issue_solver_commit_count', 'issue_solver_level', 'expert_comments', 'mid_level_comments', 'newcomer_comments',
        'openedIRinHW', 'openedIRinOther', 'openedPR', 'numberofexpertdevelopers', 'howmanyexpertdeveloperscommittedrecently', 'numberofdevelopersassignedtoissues', 
        'numberofassignedissuestoexpertdevelopers', 'expertdevelopersrecentcommitcount', 'recentPRresolvingaveragetime', 'recentIRresolvingaveragetime', 'stars_count'
    ])
        
    issue_events_folder_path = os.path.join(file_path1, 'issues_events')
    help_Wanted_date = {}
    issue_created_date = {}
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
                                            help_wanted_exists = any(label.get('name').lower() == 'help' and label.get('name').lower() == 'wanted'  for label in labels)
                                            if help_wanted_exists:
                                             #   print(url, sub_item.get('created_at'))
                                                help_Wanted_date[url] = sub_item.get('created_at')
                                        else:
                                            url = sub_item.get('url')
                                            if 'help' in sub_item.get('label').get('name').lower() and 'wanted' in sub_item.get('label').get('name').lower():
                                                help_Wanted_date[url] = sub_item.get('created_at')
                                               # print(url, sub_item.get('created_at'))
                                                
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    
    wantedBug = {}
    unWantedBug = {}
    Bugs = {}
    indx = 0
    count1 = count2 = 0
    # Iterate over each row in the DataFrame
    df = pd.read_csv(update_moreinfo_path, low_memory=False)
    for index, row in df.iterrows():
        url = row['url']
      #  print(url)
        urls, codesnippet = count_urls_and_code_snippets(str(row['body']))

        helpwantedtagcame = ''
        afterhwtagcameresolvingtime = ''
        
        if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']).lower():
            count1 += 1
            count3 += 1
            if url in help_Wanted_date:
                count2 += 1
                count4 += 1
                helpwantedtagcame = dateDifference(help_Wanted_date[url], created_at_date[url])
                afterhwtagcameresolvingtime = abs(row['resolution time'] - helpwantedtagcame)

        df1.loc[indx] = [
            row['url'],
            row['title'],
            row['body'],
            row['label'],
            codesnippet,
            urls,
            helpwantedtagcame,
            afterhwtagcameresolvingtime,
            row['prev30DaysComments'],
            row['bugnum'],
            row['latest_bugnum'],
            row['hw_ratio'],
            row['reporter_experience_level'],
            row['assigned_issue_count'],
            row['latest_assigned_issue_count'],
            row['state'],
            row['mentioned_developer'],
            row['subscriber'],
            row['lengthofthetitle'],
            row['lengthofthedesc'],
            row['commentNumber'],
            row['resolution time'],
            row['issue_solver'],
            row['issue_solver_commit_count'],
            row['issue_solver_level'],
            row['expert_comments'],
            row['mid_level_comments'],
            row['newcomer_comments'],
            row['openedIRinHW'],
            row['openedIRinOther'],
            row['openedPR'],
            row['numberofexpertdevelopers'],
            row['howmanyexpertdeveloperscommittedrecently'],
            row['numberofdevelopersassignedtoissues'],
            row['numberofassignedissuestoexpertdevelopers'],
            row['expertdevelopersrecentcommitcount'],
            row['recentPRresolvingaveragetime'],
            row['recentIRresolvingaveragetime'],
            repo_name[str(subdir).lower()]
        ]
        indx = indx + 1
            
    print(count1, count2, count3, count4)
    #df1.to_csv(savedFile, index=False, escapechar='\\')  
