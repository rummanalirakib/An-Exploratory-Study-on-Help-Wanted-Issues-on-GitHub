import re
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
from datetime import datetime, timedelta
import statistics

def dateDifference(date1, date2):
    date1 = datetime.strptime(date1, "%Y-%m-%dT%H:%M:%SZ").date()
    date2 = datetime.strptime(date2, "%Y-%m-%dT%H:%M:%SZ").date()
    date_difference = date1 - date2
    return abs(date_difference.days)

def help_wanted(val):
    return 'help' in val.lower() and 'wanted' in val.lower()

def good_first_issue(val):
    return 'good' in val.lower() and 'first' in val.lower()

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
    count = sum(start_date <= date <= end_date for date in assignee_date_list)

    return count
'''
def extract_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://[^\s,]+')
    # Find all URLs in the input text
    urls = url_pattern.findall(text)
    
    # Keywords to check in the text, with word boundaries
    keywords = ['fix', 'fixes', 'fixed', 'done', 'resolve', 'resolves', 'resolved', 'close', 'closes', 'closed', 'Pull Request', 'commit']
    keyword_patterns = [rf'\b{re.escape(keyword)}\b' for keyword in keywords]
    
    # Check if any of the keywords are present in the text (case insensitive)
    keyword_found = any(re.search(pattern, text, re.IGNORECASE) for pattern in keyword_patterns)
    
    return urls, keyword_found

def extractIssueNum(text):
    pattern = r'#(\d+)'
    issue_references = re.findall(pattern, sub_item.get('body'))
    
    # Keywords to check in the text, with word boundaries
    keywords = ['fix', 'fixes', 'fixed', 'done', 'resolve', 'resolves', 'resolved', 'close', 'closes', 'closed', 'Pull Request', 'commit']
    keyword_patterns = [rf'\b{re.escape(keyword)}\b' for keyword in keywords]
    
    # Check if any of the keywords are present in the text (case insensitive)
    keyword_found = any(re.search(pattern, text, re.IGNORECASE) for pattern in keyword_patterns)
    
    return issue_references, keyword_found
'''

def extract_urls_and_keywords(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://[^\s,]+')
    
    # Keywords to check in the text, with word boundaries
    keywords = ['fix', 'fixes', 'fixed', 'done', 'resolve', 'resolves', 'resolved', 'close', 'closes', 'closed', 'Pull Request', 'commit']
    # Pattern to match keyword followed by any characters up to the URL
    keyword_patterns = [rf'\b{re.escape(keyword)}\b.*?({url_pattern.pattern})' for keyword in keywords]
    
    # Find the URL right after a keyword
    first_url_after_keyword = None
    for pattern in keyword_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            first_url_after_keyword = match.group(1)
            break

    return first_url_after_keyword


def find_keywords_and_issues(text, keywords):
    # Create a pattern to match keywords followed by optional characters and then an issue reference
    patterns = [rf'\b{re.escape(keyword)}\b\s*#(\d+)' for keyword in keywords]
    
    # Initialize variables
    first_issue_reference = None
    
    # Search for the first issue reference after a keyword
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            first_issue_reference = match.group(1)
            break
    

    return first_issue_reference

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

def extract_issue_number(issue_path):
    # Regular expression to capture the number before any non-numeric characters
    match = re.match(r'(\d+)', issue_path)
    if match:
        return match.group(1)
    return None

# Folder path containing the CSV files
root_directory = r'C:\Study\Software Engineering Research Paper\Good First Issue\ghso-main\config\data1'

# List to store modified DataFrames
modified_dfs = []

new_folder_path = r'C:\Study\Software Engineering Research Paper\Good First Issue Reports Dataframe'
file_path = r'C:\Study\Software Engineering Research Paper\Good First Issue Initial Dataset'
issue_comment_path = r'C:\Study\Software Engineering Research Paper\Good First Issue Events and Comments'
file_path_Initial_dataset = file_path
issueSolverIssueMean = []
issueSolverCommitMean = []
count1 = count2 = count3 = count4 = 0
# Loop through each file in the folder
for subdir in os.listdir(root_directory):
    print(subdir)

    subdir_path = os.path.join(root_directory, subdir)
  #  print(subdir, subdir_path)
    
    commits_folder_path = os.path.join(subdir_path, 'commits')
    pulls_folder_path = os.path.join(subdir_path, 'pulls')
   # print('comment:', commits_folder_path)
    
    
    commentatorNumber = {}
    commentNumber = {}
    dtypes = {'labels.name': str}
    tempCsv = subdir + '.csv'
    
    file_path1 = os.path.join(issue_comment_path, tempCsv)
   # print('csv File: ', file_path1)
    try:
        df = pd.read_csv(file_path1, dtype=dtypes, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        continue
    commentator_date = {}
    hw_gfi_changed_date = {}
    for index, row in df.iterrows():
        if pd.notna(row['comments_with_date']) and row['comments_with_date'] != '':
            commentator_date[row['url']] = row['comments_with_date']
        if pd.notna(row['hw_gfi_changed_date']) and row['hw_gfi_changed_date'] != '':
            hw_gfi_changed_date[row['url']] = row['hw_gfi_changed_date']
    

    file_path1 = os.path.join(new_folder_path, tempCsv)
   # print('csv File: ', file_path1)
    try:
        df = pd.read_csv(file_path1, dtype=dtypes, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        continue
    
    
    issueUrlsCreatedDate = {}
    reporterKey = {}
    assignee_counts = {}
    assignee_dates = {}
    for index, row in df.iterrows():
        issueUrlsCreatedDate[row['url']] = row['created_at']
        reporterKey[row['url']] = row['user.login']
        assignesInfo = str(row['assignees.login'])
        date = row['created_at']
        if assignesInfo and assignesInfo != "nan":
            assignees_list = re.findall(r'<(.*?)>', assignesInfo)
            for assignee in assignees_list:
                if assignee in assignee_counts:
                    assignee_counts[assignee] += 1
                    assignee_dates[assignee].append(date)
                else:
                    assignee_counts[assignee] = 1  
                    assignee_dates[assignee] = [date]
        
    commits = {}
    maximum1 = 0
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
                                        maximum1 = max(maximum1, len(commits[committer]))
                                    else:
                                        commits[committer] = [committer_date]
                                        maximum1 = max(maximum1, 1)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
      #  print('Starting reading issue events:')
        pullIssueUrl = {}
        pullRequestInfo = {}
        maximum=0
        pullIssueInfo = {}
        for file_name in os.listdir(pulls_folder_path):
            file_path = os.path.join(pulls_folder_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as json_file:
                        item = json.load(json_file)
                        if isinstance(item, list):
                            for sub_item in item:
                                if sub_item.get('body'):
                                    keywords = ['fix', 'fixes', 'fixed', 'done', 'resolve', 'resolves', 'resolved', 'close', 'closes', 'closed', 'Pull Request', 'commit']
                                    url = extract_urls_and_keywords(sub_item.get('body'))
                                    # Loop through the URLs and check if any contain 'issues/'
                                   # pattern = r'#(\d+)'
                                    iReference = find_keywords_and_issues(sub_item.get('body'), keywords)

                                    if url is not None:
                                        if 'issues' in url:
                                            start_index = url.find('issues/') + len('issues/')
                                            issue_path = extract_issue_number(url[start_index:])
                                            user_info = sub_item['user'].get('login')
                                            created_at = sub_item.get('created_at')
                                            if user_info in pullRequestInfo:
                                                pullRequestInfo[user_info].append(created_at)
                                            else:
                                                pullRequestInfo[user_info] = [created_at]
                                            pullIssueUrl[issue_path] = user_info
                                            pullIssueInfo[issue_path] = sub_item.get('html_url')
                                        
                                    
                                    if iReference is not None:
                                            user_info = sub_item['user'].get('login')
                                            created_at = sub_item.get('created_at')
                                            if user_info in pullRequestInfo:
                                                pullRequestInfo[user_info].append(created_at)
                                            else:
                                                pullRequestInfo[user_info] = [created_at]
                                            pullIssueUrl[iReference] = user_info
                                            pullIssueInfo[iReference] = sub_item.get('html_url')
                                            
                                        
                                    
                        
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")

        savedFile = 'C:/Study/Software Engineering Research Paper/HWI Helpness Dataset Creation/HW Issues All Related Info/' + subdir + '.csv'
        
        savedFile = os.path.join(savedFile)
        df1 = pd.DataFrame(columns=[
            'url', 'title', 'body', 'label', 'label Changer', 'bugnum', 'latest_bugnum', 'commentatorNumber', 'commentNumber', 'state', 'resolution time', 'issue_label_change_to_gfi_hw', 'mentioned_developer', 'assigned_issue_count', 'latest_assigned_issue_count', 'pull_request_des', 'issue_solver', 'issue_solver_commit_count', 'issue_solver_level', 'commentator_total_bug_report', 'commentator_total_commit_done', 'unique_commentators'
        ])
        indx = 0
        Bugs = {}
        latest_bugs_dates = {}
        tempCsv = subdir + '.csv'
        file_path_Initial_dataset1 = os.path.join(file_path_Initial_dataset, tempCsv)
        print('Final csv File: ', file_path_Initial_dataset1)
        try:
            df = pd.read_csv(file_path_Initial_dataset1, dtype=dtypes, low_memory=False)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            continue
        # Iterate over each row in the DataFrame
        countHw = 0
        for index, row in df.iterrows():
            string_to_label = str(row['label']).lower();
            if string_to_label == '' and string_to_label == 'nan':
                continue
            if 'help' in row['label'] and 'wanted' in row['label']:
                countHw += 1
            url = row['url']
            date = issueUrlsCreatedDate[row['url']]
            
            key = str(reporterKey[url])
            Bugs[key] = Bugs.get(key, 0) + 1
            if key in latest_bugs_dates:
                latest_bugs_dates[key].append(date)
            else:
                latest_bugs_dates[key] = [date]


            count1 = count1 + 1
            
            labelChanger = row['label Changer']
            commentatorNumber = row['commentatorNumber']
            commentNumber = row['commentNumber']
            
            issue_body = str(row['body'])
            if labelChanger and labelChanger != "nan" and labelChanger in assignee_counts:
                assigned_issues = assignee_counts[labelChanger]
                latest_assigned_issues = count_dates_within_range(labelChanger, date, assignee_dates, 30)
            else:
                assigned_issues = 0
                latest_assigned_issues = 0
            

            mentioned_developer_count = row['mentioned_developer']
            resolutionTime = row['resolution time']
            issueLabelChange = row['issue_label_change_to_gfi_hw']
            count2 = count2 + 1
            bugCount = 0
            latestbugCount = 0
            if labelChanger in Bugs:
                bugCount = Bugs[labelChanger]
                latestbugCount = count_dates_within_range(labelChanger, date, latest_bugs_dates, 30)
            
            start_index = url.find('issues/') + len('issues/')
            issue_path = url[start_index:]
            issueSolver = ''
            issueSolverCommitCount = -1
            issueSolverInformationPull = ''
            if issue_path in pullIssueUrl:
                issueSolver  = pullIssueUrl[issue_path]
                issueSolverInformationPull = pullIssueInfo[issue_path]
                count3+=1
                if issueSolver in commits:
                    issueSolverCommitCount = count_dates_within_range(issueSolver, date, commits, 10000)
                    issue_solver_level = committerLevelDecide(date, commits, issueSolver, issueSolverCommitCount)
                    count4+=1
            
            if issueSolver == '':
                continue
            commentatorBugReport = -1
            commentatorCommitNumbers = -1
            commentatorIssueSolvedCount = -1
            uniqueCommentatorLength = -1
            
            if row['url'] in hw_gfi_changed_date:
                commentatorBugReport = 0
                commentatorCommitNumbers = 0
                commentatorIssueSolvedCount = 0
                uniqueCommentatorLength = 0
                
            if row['url'] in hw_gfi_changed_date and row['url'] in commentator_date:
                given_date_str = hw_gfi_changed_date[row['url']]

                date_name_pattern = r'<([^:]+):(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)>'  
                matches = re.findall(date_name_pattern, commentator_date[row['url']])
                dates_and_names = [(name, date_str) for name, date_str in matches]
                
                names_after_given_date = [name for name, date in dates_and_names if date > given_date_str]
                
                unique_names_after_given_date = set(names_after_given_date)
                uniqueCommentatorLength = len(unique_names_after_given_date)
                
                for name, date_str in matches:
                    if name in Bugs:
                        commentatorBugReport += Bugs[name]
                        
                    if issue_path in pullIssueUrl:
                        issueSolver1  = name
                        if issueSolver1 in commits:
                            commentatorCommitNumbers += count_dates_within_range(issueSolver, given_date_str, commits, 10000)
                
            df1.loc[indx] = [
                row['url'], row['title'], row['body'] , row['label'] , labelChanger, bugCount, latestbugCount, commentatorNumber, commentNumber, row['state'], resolutionTime, issueLabelChange, mentioned_developer_count, assigned_issues, latest_assigned_issues, issueSolverInformationPull, issueSolver, issueSolverCommitCount, issue_solver_level, commentatorBugReport, commentatorCommitNumbers, uniqueCommentatorLength
            ]
            indx = indx + 1
                
        print(count1, count2, count3, count4)
        if countHw != 0:
            df1.to_csv(savedFile, index=False, escapechar='\\')     
