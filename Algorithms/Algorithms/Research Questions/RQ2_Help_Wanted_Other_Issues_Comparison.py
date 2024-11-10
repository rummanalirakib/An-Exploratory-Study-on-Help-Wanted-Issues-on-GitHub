import json
import requests
import os
import time
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
final_github_access_token = ""
count1=0
timeCount = {}

def dateDifference(date1, date2):
    print(date1, date2)
    date1 = datetime.strptime(date1, "%Y-%m-%dT%H:%M:%SZ").date()
    date2 = datetime.strptime(date2, "%Y-%m-%dT%H:%M:%SZ").date()
    date_difference = date1 - date2
    return abs(date_difference.days)

def check_rate_limit(token, events_url, count1, count2, count3):
    headers = {
        'Authorization': f'Bearer {token}',
    }
    url = 'https://api.github.com/rate_limit'
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        rate_limit_info = response.json()
        
        if rate_limit_info:
            core_remaining = rate_limit_info['resources']['core']['remaining']
            print(core_remaining, events_url, count1, count2, count3)
            return core_remaining
        
        return 0
    except requests.exceptions.RequestException as e:
        print(f"Error checking rate limit: {e}")
        return 0

def get_issue_events(api_url, access_token):
    global count1, final_github_access_token
    headers = {
        'Authorization': f'Bearer {access_token}',
    }

    response = requests.get(api_url, headers=headers, timeout=30)

    if response.status_code == 200:
        try:
            data = response.json()
            return data
        except json.JSONDecodeError:
            print("Error decoding JSON in response:", response.text)
            return None
        

def process_and_write_events(events_url, github_access_token):
    issue_events = get_issue_events(events_url, github_access_token)
  #  print("Events: " + events_url)
    subscribed = 0
    if issue_events is not None:
        for event in issue_events:
            if event is not None:        
                if 'subscribed' in event.get('event'):
                    actor = event.get('actor')
                    if actor is None:
                        continue
                    else:
                        subscribed += 1
    
    return subscribed
                      

def process_and_write_comments(comments_url, github_access_token, issue_url):
    issue_comments = get_issue_events(comments_url, github_access_token)
  #  print("Events: " + events_url)
    commentatorNumber = {}
    commentNumber = {}
    if issue_comments is not None:
        for comment in issue_comments:
            commentator = comment.get('user').get('login')
            if issue_url in commentatorNumber:
                if commentator not in commentatorNumber[issue_url]:
                    commentatorNumber[issue_url] += ',' + commentator
            else:
                commentatorNumber[issue_url] = commentator
            commentNumber[issue_url] = commentNumber.get(issue_url, 0) + 1
        
        if issue_url in commentatorNumber:
            return commentatorNumber[issue_url], commentNumber[issue_url]
    return -1, -1

# Folder path containing the CSV files
folder_path = r'C:\Study\Software Engineering Research Paper\Good First Issue Reports Dataframe'

# List to store modified DataFrames
modified_dfs = []
filenames = os.listdir(folder_path)
count1 = count2 = count3 = 0
gfi = hw = othr = 0
title = []
body = []
label = []
keyword_counts = {}
events_URL = []
created_at = []

final_github_access_token = ""

# Loop through each file in the folder
for filename in filenames:
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file and specify the data types for columns
        dtypes = {'labels.name': str}  # Add other columns if necessary
        df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
        print(filename)
        # Specify the column name you want to check
        column_name_to_check = 'labels.name'
        
        # Specify the value you want to check for in the specified column
        value_to_check = 'wontfix'
        savedFile = 'C:/Study/Software Engineering Research Paper/GFI and HW other issue related information/' + filename
        
        savedFile = os.path.join(savedFile)
        df1 = pd.DataFrame(columns=[
            'url', 'title', 'body', 'label', 'commentatorNumber', 'commentNumber', 'state', 'resolution time', 'mentioned_developer', 'subscribed'
        ])
        indx = 0
        Bugs = {}
        count4 = 0
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            if count4 >= 500:
                break
            string_to_split = str(row['labels.name'])
            pullRequest = str(row['html_url'])
            if 'pull' in pullRequest:
                continue
                
            key = str(row['user.login'])
            Bugs[key] = Bugs.get(key, 0) + 1
            if ('help' in string_to_split.lower() and 'wanted' in string_to_split.lower()):
                continue
            if string_to_split and string_to_split != "nan" and 'closed' in row['state'] and (('bug' in string_to_split.lower()) or ('enhancement' in string_to_split.lower()) or ('question' in string_to_split.lower())):
                print(string_to_split, row['events_url'])
                labelLower = str(row['labels.name']).lower()
                events_url = row['events_url']
                comments_url = row['comments_url']
                count1 = count1 + 1
                while check_rate_limit(final_github_access_token, events_url, count1, count2, count4) <= 2:
                    time.sleep(10)
                count2 += 1
                print(process_and_write_events(events_url, final_github_access_token))
                subscribed = process_and_write_events(events_url, final_github_access_token)
                commentatorNumber, commentNumber = process_and_write_comments(comments_url, final_github_access_token, row['url'])
                issue_body = str(row['body'])
                if issue_body is None:
                    mentioned_developer_count = 0
                else:
                    mentioned_developer_count = issue_body.count('@')
                resolutionTime = dateDifference(row['created_at'], row['closed_at'])

                df1.loc[indx] = [
                    row['url'], row['title'], row['body'] , row['labels.name'], commentatorNumber, commentNumber, row['state'], resolutionTime, mentioned_developer_count, subscribed
                ]
                indx = indx + 1
                count4 = count4+ 1

                
        print(count1, count2)
        df1.to_csv(savedFile, index=False, escapechar='\\')  
