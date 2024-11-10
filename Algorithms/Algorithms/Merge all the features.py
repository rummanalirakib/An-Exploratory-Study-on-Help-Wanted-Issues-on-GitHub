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



# List to store modified DataFrames
modified_dfs = []
count3 = count4 = 0
title = []
body = []
label = []
keyword_counts = {}
Reporter_Experience = []
count1 = count2 = 0
indexing = 0
allIndex = 0

# Loop through each file in the folder
root_directory = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Data'
csv_file_folder = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Dataset Final Preparation for Machine Learning'
another_folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Business of developers Calculation'
saved_folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Machine Learning Dataset For All Features'

# Loop through each file in the folder
for subdir in os.listdir(root_directory):
    print(subdir)
    filename = subdir + '.csv'
    file_path = os.path.join(csv_file_folder, filename)
    file_path1 = os.path.join(another_folder_path, filename)
    indexing = indexing + 1
    # Read the CSV file and specify the data types for columns
    dtypes = {'labels.name': str}  # Add other columns if necessary    
    
    mappedUrl = {}
    df = pd.read_csv(file_path1, dtype=dtypes, low_memory=False)
    openedIRinHW = {}
    openedIRinOther = {}
    openedPR = {}
    number_of_expert_developers = {}
    expert_devs_committed_recently = {}
    number_of_assigned_devs = {}
    assigned_issues_to_expert_devs = {}
    expert_devs_recent_commit_count = {}
    avg_PR_resolve_time = {}
    avg_IR_resolve_time = {}
    
    # Loop through the rows and map the features based on 'url'
    for index, row in df.iterrows():
        url = row['url']
        
        # Assign values to the dictionaries based on 'url'
        openedIRinHW[url] = row['openedIRinHW']
        openedIRinOther[url] = row['openedIRinOther']
        openedPR[url] = row['openedPR']
        number_of_expert_developers[url] = row['numberofexpertdevelopers']
        expert_devs_committed_recently[url] = row['howmanyexpertdeveloperscommittedrecently']
        number_of_assigned_devs[url] = row['numberofdevelopersassignedtoissues']
        assigned_issues_to_expert_devs[url] = row['numberofassignedissuestoexpertdevelopers']
        expert_devs_recent_commit_count[url] = row['expertdevelopersrecentcommitcount']
        avg_PR_resolve_time[url] = row['recentPRresolvingaveragetime']
        avg_IR_resolve_time[url] = row['recentIRresolvingaveragetime']
        
    
    df1 = pd.DataFrame(columns=[
        'url', 'title', 'body', 'label', 'prev30DaysComments', 'bugnum', 'latest_bugnum', 'hw_ratio', 'reporter_experience_level', 
        'assigned_issue_count', 'latest_assigned_issue_count', 'state', 'mentioned_developer', 
        'subscriber', 'lengthofthetitle', 'lengthofthedesc', 'commentNumber', 'resolution time',
        'issue_solver', 'issue_solver_commit_count', 'issue_solver_level', 'expert_comments', 'mid_level_comments', 'newcomer_comments',
        'openedIRinHW', 'openedIRinOther', 'openedPR', 'numberofexpertdevelopers', 'howmanyexpertdeveloperscommittedrecently', 'numberofdevelopersassignedtoissues', 
        'numberofassignedissuestoexpertdevelopers', 'expertdevelopersrecentcommitcount', 'recentPRresolvingaveragetime', 'recentIRresolvingaveragetime'
    ])
    
    savedFile = os.path.join(saved_folder_path, filename)
    
    indx = 0
    df = pd.read_csv(file_path, low_memory=False)
    for index, row in df.iloc[::-1].iterrows():
        url = row['url']
        reporterlevel = 0
        if 'mid' in row['reporterlevel']:
            reporterlevel = 1
        elif 'expert' in row['reporterlevel']:
            reporterlevel = 2
            
        df1.loc[indx] = [
            row['url'],
            row['title'],
            row['body'],
            row['label'],
            row['prev30DaysComments'],
            row['bugnum'],
            row['latest_bugnum'],
            row['hw_ratio'],
            row['reporterlevel'],
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
            openedIRinHW[url],
            openedIRinOther[url],
            openedPR[url],
            number_of_expert_developers[url],
            expert_devs_committed_recently[url],
            number_of_assigned_devs[url],
            assigned_issues_to_expert_devs[url],
            expert_devs_recent_commit_count[url],
            avg_PR_resolve_time[url],
            avg_IR_resolve_time[url]
        ]
        indx = indx + 1
        count1 += 1
            
    df1.to_csv(savedFile, index=False, escapechar='\\') 
            
    print(count1)
        
