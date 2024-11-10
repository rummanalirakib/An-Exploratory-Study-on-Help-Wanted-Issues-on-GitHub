import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Folder paths containing the CSV files
folder_path = r'C:\Study\Software Engineering Research Paper\GFI and HW other issue related information'
folder_path1 = r'C:\Study\Software Engineering Research Paper\Good First Issue Fifth Dataset'
folder_path2 = r'C:\Study\Software Engineering Research Paper\GFI and HW other issue related information'
data = {
    'Issue Type': [],
    'Days_Resolution': [],
    'Actors_Subscription': [],
    '@mentioned': [],
    'Commenter': [],
    'Comments': []
}
subscribers = {}
# Loop through each file in the folder
for filename in os.listdir(folder_path):
    print(filename)
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Load the CSV file
        dtypes = {'label': str}  
        df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
        
        # Check if 'label' column exists
        if 'label' not in df.columns:
            print(f"Skipping file {filename} as it does not contain 'label' column.")
            continue
        
        for index, row in df.iterrows():
            data['Issue Type'].append('Other Issue')
            if row['commentatorNumber'] != -1:
                data['Commenter'].append(len(row['commentatorNumber'].split(',')))
            else:
                data['Commenter'].append(0)
            if row['commentNumber'] != -1:
                data['Comments'].append(row['commentNumber'])
            else:
                data['Comments'].append(0)
            data['Days_Resolution'].append(row['resolution time'])
            data['@mentioned'].append(row['mentioned_developer'])
            data['Actors_Subscription'].append(row['subscribed'])
            
for filename in os.listdir(folder_path2):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path2, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Load the CSV file
        dtypes = {'label': str}  
        df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
        
        for index, row in df.iterrows():
            if 'help' in row['label'] and 'wanted' in row['label']:
                subscribers[row['url']] = row['subscribed']
                
for filename in os.listdir(folder_path1):
    print(filename)
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path1, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Load the CSV file
        dtypes = {'label': str}  
        df = pd.read_csv(file_path, dtype=dtypes, low_memory=False)
        
        for index, row in df.iterrows():
            if 'help' in row['label'] and 'wanted' in row['label']:
                data['Issue Type'].append('HWIs')
                if row['commentatorNumber'] != -1:
                    data['Commenter'].append(len(row['commentatorNumber'].split(',')))
                else:
                    data['Commenter'].append(0) 
                   
                if row['commentNumber'] != -1:
                    data['Comments'].append(row['commentNumber'])
                else:
                    data['Comments'].append(0)
                    
                data['Days_Resolution'].append(row['resolution time'])
                data['@mentioned'].append(row['mentioned_developer'])
                if row['url'] in subscribers:
                    data['Actors_Subscription'].append(subscribers[row['url']])
                else:
                    data['Actors_Subscription'].append(0)

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Calculate min, max, mean, median for each metric and issue type
metrics = ['Days_Resolution', 'Actors_Subscription', '@mentioned', 'Commenter', 'Comments']
statistics = df.groupby('Issue Type')[metrics].agg(['min', 'max', 'mean', 'median'])

# Print the statistics
for metric in metrics:
    print(f"\nStatistics for {metric}:\n")
    print(statistics[metric])

# Plotting the data
plt.figure(figsize=(15, 10))

# Plot each metric as a boxplot
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Issue Type', y=metric, data=df)
    plt.title(metric)

plt.tight_layout()
plt.show()
