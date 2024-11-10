import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import textstat
import numpy as np
from datetime import datetime
import networkx as nx
import random


# Function to preprocess text
def preprocess_text(raw_text):
    # Convert to lowercase
    lowercase_text = raw_text.lower()

    # Remove punctuation
    cleaned_text = re.sub(r'[^\w\s]', '', lowercase_text)

    # Tokenize the text
    tokens = word_tokenize(cleaned_text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Join the tokens back into a preprocessed text
    preprocessed_text = ' '.join(stemmed_tokens)

    return preprocessed_text

def extract_technical_information(description):

    # Regular expressions for recognition
    stack_regex = re.compile(r'\btrace\b', re.IGNORECASE)
    patch_regex = re.compile(r'\b(?:fix|patch)\b', re.IGNORECASE)
    testcase_regex = re.compile(r'\btest\s*case\b', re.IGNORECASE)
    screenshot_regex = re.compile(r'\b(?:window|view|screenshot)\b', re.IGNORECASE)
    steps_to_reproduce_patterns = [r'\bsteps?\s+to\s+reproduce\b', r'\breproduce\s+steps\b']
    code_constructs = ['class', 'function', 'if', 'else', 'for', 'while']

    completeness_metrics = []

    # Iterate through each description in the list
    has_stack = False
    has_step = False
    has_code = False
    has_patch = False
    has_testcase = False
    has_screenshot = False
    # Check for stack traces
    if re.search(stack_regex, description):
        has_stack = True

    # Check for patches
    if re.search(patch_regex, description):
        has_patch = True

    # Check for test cases
    if re.search(testcase_regex, description):
        has_testcase = True

    # Check for screenshots
    if re.search(screenshot_regex, description):
        has_screenshot = True

    # Check for steps to reproduce
    for pattern in steps_to_reproduce_patterns:
        if re.search(pattern, description):
            has_step = True
            break

    # Check for code samples
    if any(construct in description for construct in code_constructs):
        has_code = True

    return has_stack, has_patch, has_testcase, has_screenshot, has_step, has_code


def dateDifference(date1, date2):
    # Input date strings
    date_str1 = date1
    date_str2 = date2

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

def largestConnectedComponent(G, reporterKey):
    # Ensure the graph is undirected
    G_undirected = G.to_undirected()
    
    # Find connected components
    connected_components = list(nx.connected_components(G_undirected))

    # Check if there are connected components
    if not connected_components:
        return 0  # No connected components

    # Find the largest connected component
    largest_connected_component = max(connected_components, key=len)

    # Check if the reporterKey is in the largest connected component
    return 1 if reporterKey in largest_connected_component else 0

def ego_betweenness_centrality(G, node):
    # Create the ego network for the specified node
    ego_network = nx.ego_graph(G, node)

    # Calculate betweenness centrality for the ego network
    betweenness_centrality = nx.betweenness_centrality(ego_network, normalized=True)

    # Return the betweenness centrality value for the node itself
    return betweenness_centrality.get(node, 0)

# Folder path containing the CSV files
folder_path = r'F:\Research\Final Help Wanted Research Datset and and Code\Including Year And All Features Prepared For ML'

all_csv_files = r'F:\Research\Final Help Wanted Research Datset and and Code\HW Issues in csv file'                
# List to store modified DataFrames
modified_dfs = []
filenames = os.listdir(folder_path)
count1 = count2 = 0

final_help_wanted = []
final_other_issue = []
# Loop through each file in the folder
for filename in filenames:
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(filename)

        another_file_path = os.path.join(all_csv_files, filename)
        df = pd.read_csv(another_file_path, on_bad_lines='skip', low_memory=False)
        reporter_info = {}
        year_info = {}
        for index, row in df.iterrows():
            dt = datetime.strptime(row['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            year = dt.year
            reporter_info[row['url']] = row['user.login']
            year_info[row['url']] = year

        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)

        df = df.sort_values(by='url')
        # Iterate over each row in the DataFrame
        help_wanted_issues = []
        other_issues = []
        for index, row in df.iterrows():
            string_to_label = str(row['label']).lower()
            if 'bug' in string_to_label or 'enhancement' in string_to_label or ('feature' in string_to_label and 'request' in string_to_label) or ('help' in string_to_label and 'wanted' in string_to_label):
               
                reporterKey = str(reporter_info[row['url']])
                commentUsers = str(row['prev30DaysComments'])
                
                reporterlvl = 0
                if row['reporter_experience_level'] == 'mid':
                    reporterlvl = 1
                elif row['reporter_experience_level'] == 'expert':
                    reporterlvl = 2
                
                separetedCommentUser = commentUsers.split(',')
                string_to_split = str(row['label']).lower()
    
                Grph = nx.DiGraph()
                
                 # Split and process each interaction in the 30-day comment history
                for commentUserKey in commentUsers.split(','):
                    reporterCommenter = commentUserKey.split(":")
                    if len(reporterCommenter) >= 2:
                       # print(reporterCommenter[1],  reporterCommenter[0])
                        Grph.add_edge(reporterCommenter[1], reporterCommenter[0])
                
                in_degree_value = out_degree_value = total_degree_value = 0
                if reporterKey in Grph:
                    in_degree_value = Grph.in_degree[reporterKey]
                    out_degree_value = Grph.out_degree[reporterKey]
                    total_degree_value = in_degree_value + out_degree_value
    
                # Centrality and other metrics calculations
                closeness_centrality = nx.closeness_centrality(Grph)
                betweenness_centrality = nx.betweenness_centrality(Grph)
                clustering_coefficient = nx.clustering(Grph.to_undirected())
                Grph.remove_edges_from(nx.selfloop_edges(Grph))
                k_coreness = nx.core_number(Grph)
                
                # Append and print each metric based on the reporter's presence in the network
                closeness_value = closeness_centrality.get(reporterKey, 0)
                betweenness_value = betweenness_centrality.get(reporterKey, 0)
                clustering_value = clustering_coefficient.get(reporterKey, 0)
                k_core_value = k_coreness.get(reporterKey, 0)
    
                # LCC Calculation
                lcc_value = largestConnectedComponent(Grph, reporterKey) if reporterKey in Grph else 0
    
                # Ego Betweenness Centrality Calculation
                ego_betweenness_value = ego_betweenness_centrality(Grph, reporterKey) if reporterKey in Grph else 0
                title_and_body = str(row['title']) + " " + str(row['body'])
                flesch = textstat.flesch_reading_ease(title_and_body)
                fog = textstat.gunning_fog(title_and_body)
                lix = textstat.lix(title_and_body)
                kincaid = textstat.flesch_kincaid_grade(title_and_body)
                ari = textstat.automated_readability_index(title_and_body)
                coleman_liau = textstat.coleman_liau_index(title_and_body)
                smog = textstat.smog_index(title_and_body)
                has_stack, has_patch, has_testcase, has_screenshot, has_step, has_code = extract_technical_information(title_and_body)
                issue_metrics = {
                    'has_stack': 0 if has_stack == 0 else 1,
                    'has_patch':  0 if has_patch == 0 else 1,
                    'has_testcase':  0 if has_testcase == 0 else 1,
                    'has_screenshot':  0 if has_screenshot == 0 else 1,
                    'has_step':  0 if has_step == 0 else 1,
                    'has_code': 0 if has_code == 0 else 1,
                    'flesch': flesch,
                    'fog': fog,
                    'lix': lix,
                    'kincaid': kincaid,
                    'ari': ari,
                    'coleman_liau': coleman_liau,
                    'smog': smog,
                    'numberofbugs': row['bugnum'],  
                    'recentnumberofbugs': row['latest_bugnum'],
                    'hw_ratio': row['hw_ratio'],
                    'ReporterExperienceNewcomer': 0 if reporterlvl == 0 else 1,
                    'ReporterExperienceOther': 0 if reporterlvl == 1 else 1,
                    'ReporterExperienceExpert': 0 if reporterlvl == 2 else 1,
                    'in_degree': in_degree_value,
                    'out_degree': out_degree_value,
                    'total_degree': total_degree_value,
                    'closeness_centrality': closeness_value,
                    'betweenness_centrality': betweenness_value,
                    'clustering_coefficient': clustering_value,
                    'k_core_value': k_core_value,
                    'lcc_value': lcc_value,
                    'ego_betweenness_value': ego_betweenness_value,
                    'openedIRinHW': row['openedIRinHW'], 
                    'openedIRinOther': row['openedIRinOther'], 
                    'openedPR': row['openedPR'], 
                    'numberofexpertdevelopers': row['numberofexpertdevelopers'],
                    'howmanyexpertdeveloperscommittedrecently': row['howmanyexpertdeveloperscommittedrecently'], 
                    'numberofdevelopersassignedtoissues': row['numberofdevelopersassignedtoissues'],
                    'numberofassignedissuestoexpertdevelopers': row['numberofassignedissuestoexpertdevelopers'],
                    'expertdevelopersrecentcommitcount': row['expertdevelopersrecentcommitcount'], 
                    'recentPRresolvingaveragetime': row['recentPRresolvingaveragetime'], 
                    'recentIRresolvingaveragetime': row['recentIRresolvingaveragetime']
                }
                
                if 'help' in string_to_split or "wanted" in string_to_split:
                     help_wanted_issues.append(issue_metrics)
                else:
                     other_issues.append(issue_metrics)

    for help_wanted_issue in help_wanted_issues:
        # Save the original help_wanted_issue in final_help_wanted
        final_help_wanted.append(help_wanted_issue)
        
        # Randomly select a maximum of 100 'Other' issues (if available)
        random_other_issues = random.sample(other_issues, min(10, len(other_issues)))
        
        # Calculate the mean for the selected 'Other' issues
        means = {key: np.median([other[key] for other in random_other_issues]) for key in help_wanted_issue}
        
        # Save the means in final_other_issue
        final_other_issue.append(means)
        print(len(final_help_wanted), len(final_other_issue), min(10, len(other_issues)))

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from math import sqrt

# Convert lists of dictionaries to DataFrames
help_wanted_df = pd.DataFrame(final_help_wanted)
other_issue_df = pd.DataFrame(final_other_issue)

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
output_path = "/home/rumman/Desktop/Research/feature_statistics.csv"
stats_df.to_csv(output_path, index=False)

print(f"Statistics saved to {output_path}")
