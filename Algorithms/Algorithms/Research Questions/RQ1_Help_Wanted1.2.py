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
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

# Folder path containing the CSV files
new_folder_path = r'D:\Final Help Wanted Research Datset and and Code\HW Issues in csv file'
root_directory = new_folder_path

count1 = count2 = 0
project_wise_help_wanted_issue = {}
project_wise_other_issue = {}
# Loop through each file in the folder
data = {
    'project': [],
    'num_hwis': [],
    'total_issues': []
}
for subdir in os.listdir(root_directory):

    subdir_path = os.path.join(root_directory, subdir)
    print(subdir, subdir_path)

    indx = 0
    Bugs = {}
    latest_bugs_dates = {}
    tempCsv = subdir
    dtypes = {'labels.name': str}
    file_path_Initial_dataset1 = os.path.join(new_folder_path, tempCsv)
    try:
        df = pd.read_csv(file_path_Initial_dataset1, dtype=dtypes, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        continue
    # Iterate over each row in the DataFrame
    project_wise_help_wanted_issue[subdir] = 0
    project_wise_other_issue[subdir] = 0
    for index, row in df.iterrows():
        string_to_split = str(row['labels.name'])
      #  if not string_to_split or string_to_split == "nan":
       #     continue
        if row['state'] == 'closed' or  row['state'] == 'open':
            # Convert to string and then check for 'help' and 'wanted'
            labels = str(row['labels.name']).lower()
            if 'help' in labels and 'wanted' in labels:
                    project_wise_help_wanted_issue[subdir] += 1

            project_wise_other_issue[subdir] += 1
    print(project_wise_help_wanted_issue[subdir], project_wise_other_issue[subdir])      
    if project_wise_other_issue[subdir] == 0:
        continue  
    if project_wise_help_wanted_issue[subdir] == 0:
        count2 += 1
    count1 += 1          
    data['project'].append(subdir)
    data['num_hwis'].append(project_wise_help_wanted_issue[subdir])
    data['total_issues'].append(project_wise_other_issue[subdir])

print(count1, count2)                    
# Merge the data into a DataFrame
project_data = pd.DataFrame(data)

# Calculate HWI ratios
project_data['HWI_Ratio'] = project_data['num_hwis'] / project_data['total_issues']

# Handle log scale visualization
project_data['HWIs_log'] = np.log10(project_data['num_hwis'] + 1)
project_data['HWI_Ratio_log'] = np.log10(project_data['HWI_Ratio'] + 1e-6)  # Add a small offset to avoid log(0)

# Violin Plot for HWIs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.violinplot(data=project_data[['HWIs_log']], inner="point", scale="width", color='white')
plt.ylabel('Log10(HWIs)')
plt.title('Distribution of HWIs per Project')

# Annotate median value
median_hwis = np.median(project_data['num_hwis'])
plt.text(0.5, np.log10(median_hwis + 1), f'Median: {median_hwis}', ha='center', va='bottom', color='black')

# Violin Plot for HWI Ratios
plt.subplot(1, 2, 2)
sns.violinplot(data=project_data[['HWI_Ratio_log']], inner="point", scale="width", color='white')
plt.ylabel('Log10(HWI Ratios)')
plt.title('Distribution of HWI Ratios per Project')

# Annotate median value
median_hwi_ratio = np.median(project_data['HWI_Ratio'])
plt.text(0.5, np.log10(median_hwi_ratio + 1e-6), f'Median: {median_hwi_ratio:.2f}', ha='center', va='bottom', color='black')

plt.tight_layout()
plt.savefig('violin_graph.png', dpi=1200)
plt.show()

# Calculate the median value of num_hwis
median_num_hwis = np.median(project_data['num_hwis'])
print("Median value of num_hwis:", median_num_hwis)