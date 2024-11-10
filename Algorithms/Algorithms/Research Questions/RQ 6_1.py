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
import numpy as np
import random

# List to store modified DataFrames
modified_dfs = []
count3 = count4 = 0
title = []
body = []
label = []
keyword_counts = {}
Reporter_Experience = []
count5 = 0
indexing = 0
allIndex = 0

# Initialize counters
count1 = count5 = count6 = 0
count2 = count3 = count4 = 0
Url_commit_count_HW = {}
Url_commit_count_other = {}

# Loop through each file in the folder
csv_file_folder1 = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Another Analysis Dataset For RQ'
csv_file_folder = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Machine Learning Dataset Preparation For All Features Final'
main_csv_folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/HW Issues in csv file'

for subdir in os.listdir(csv_file_folder):
    print(subdir)
    if 'eact-bootstrap-table' in subdir:
        subdir = 'allenfang_react-bootstrap-table.csv'
    file_path = os.path.join(csv_file_folder, subdir)

    df = pd.read_csv(file_path, low_memory=False)
    
    for index, row in df.iterrows():
        url = row['url']
        if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']).lower():
            if str(row['issue_solver_commit_count']) != 'nan':
                Url_commit_count_HW[url] = True

        else:
            if str(row['state']) == 'closed' and str(row['issue_solver_commit_count']) != 'nan':
                Url_commit_count_other[url] = True
                
count4 = count5 = count6 = 0
for subdir in os.listdir(csv_file_folder1):
    print(subdir)
    file_path = os.path.join(csv_file_folder1, subdir)
    
    df = pd.read_csv(file_path, low_memory=False)
    
    for index, row in df.iterrows():
        url = row['url']
        if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']).lower() and str(row['commitExist']) == 'True':
            Url_commit_count_HW[url] = True

        else:
            if str(row['state']) == 'closed' and str(row['commitExist']) == 'True':
                Url_commit_count_other[url] = True

subdir_wise_1_HW = {}     
subdir_wise_1_other = {}  
subdir_wise_2_HW = {}
subdir_wise_2_other = {}  
project = 0          
for subdir in os.listdir(main_csv_folder_path):
    print(subdir)
    file_path = os.path.join(main_csv_folder_path, subdir)
    
    df = pd.read_csv(file_path, low_memory=False)
    count7 = count8 = count9 = 0
    count10 = count11 = count12 = 0
    
    for index, row in df.iterrows():
        url = row['url']
        if str(row['labels.name']) == 'nan' and 'pull':
            continue
        
        if 'help' in str(row['labels.name']).lower() and 'wanted' in str(row['labels.name']).lower():
            count1 += 1
            count7 += 1
            if url in Url_commit_count_HW and row['state'] == 'closed':
                count2 += 1
                count8 += 1
            if row['state'] == 'closed':
                count3 += 1
                count9 += 1
        else:
            count4 += 1
            count10 += 1
            if url in Url_commit_count_other and row['state'] == 'closed':
                count5 += 1
                count11 += 1
            if row['state'] == 'closed':
                count6 += 1
                count12 += 1
                
    print(count7, count8, count9, count10)
    project += 1
    project_name = "P" + str(project)
    subdir_wise_1_HW[project_name] = count8 / count7
    subdir_wise_2_HW[project_name] = count9 / count7
    subdir_wise_1_other[project_name] = count11 / count10
    subdir_wise_2_other[project_name] = count12 / count10

print(count1, count2, count3, count4, count5, count6)
print('Percentage: ', count2/count1, count3/count1, count5/count4, count6/count4)
import matplotlib.pyplot as plt
import numpy as np

# Prepare the data for plotting
subdirs = list(subdir_wise_1_HW.keys())
values_1_HW = list(subdir_wise_1_HW.values())
values_1_other = list(subdir_wise_1_other.values())
values_2_HW = list(subdir_wise_2_HW.values())
values_2_other = list(subdir_wise_2_other.values())

# Set up bar positions and width
x = np.arange(len(subdirs))  # the label locations
width = 0.35  # the width of the bars

# Create subplots for two comparisons
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot 1: subdir_wise_1_HW vs subdir_wise_1_other
rects1 = ax1.bar(x - width/2, values_1_HW, width, label='HW Issues')
rects2 = ax1.bar(x + width/2, values_1_other, width, label='Other Issues')

# Add labels, title, and custom x-axis tick labels, etc.
ax1.set_xlabel('Pseudo Project Name')
ax1.set_ylabel('Ratio')
ax1.set_title('Comparison of HW and Other Issues Considering Only Closed By Commit')
ax1.set_xticks(x)
ax1.set_xticklabels(subdirs, rotation=90, ha='right')
ax1.legend()

# Set x-axis to start from the left edge
ax1.set_xlim(-0.5, len(subdirs) - 0.5)

# Plot 2: subdir_wise_2_HW vs subdir_wise_2_other
rects3 = ax2.bar(x - width/2, values_2_HW, width, label='HW Issues')
rects4 = ax2.bar(x + width/2, values_2_other, width, label='Other Issues')

ax2.set_xlabel('Pseudo Project Name')
ax2.set_ylabel('Ratio')
ax2.set_title('Comparison of HW and Other Issues Considering Only Closed')
ax2.set_xticks(x)
ax2.set_xticklabels(subdirs, rotation=90, ha='right')
ax2.legend()

# Set x-axis to start from the left edge
ax2.set_xlim(-0.5, len(subdirs) - 0.5)

# Adjust layout to ensure everything fits without overlap
plt.tight_layout()

# Show the plots
plt.show()
