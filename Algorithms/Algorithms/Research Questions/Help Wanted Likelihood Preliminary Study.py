import os
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

def percent_formatter(x, pos):
    return f'{x:.0f}%'

# Initialize the dictionary to store the solving likelihood
Projectwise_solving_likelihood = {}
project_name_mapping = {}

# Folder path containing the CSV files
folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/HW Issues in csv file'
count3 = 0
count4 = count5 = 0
count6 = count7 = 0
repository_stars = {}


# Loop through each file in the folder
for subdir in os.listdir(folder_path):
    print(subdir)
    # Load the CSV file
    file_path = os.path.join(folder_path, subdir)
    if not os.path.exists(file_path):
        continue
    df = pd.read_csv(file_path, low_memory=False)
    count1 = count2 = 0
    for index, row in df.iterrows():
        string_to_label = str(row['labels.name']).lower()
        if string_to_label == '' or string_to_label == 'nan' or 'pull' in str(row['pull_request.html_url']):
            continue
        if 'help' in string_to_label and 'wanted' in string_to_label:
            count2 += 1
            if row['state'] == 'closed':
                count1 += 1
        else:
            count5 += 1
            if row['state'] == 'closed':
                count4 += 1

    if count2 != 0:
        count3 += 1
        project_name = 'P' + str(count3)
        Projectwise_solving_likelihood[project_name] = (count1 / count2)*100
        project_name_mapping[project_name] = subdir
        count6 += count1
        count7 += count2

print(count6/count7, count4/count5)

# Sort Projectwise_solving_likelihood based on repository names
sorted_project_names = sorted(Projectwise_solving_likelihood.keys(), key=lambda x: project_name_mapping[x])
sorted_Projectwise_solving_likelihood = {project: Projectwise_solving_likelihood[project] for project in Projectwise_solving_likelihood}

# Calculate percentages for solving likelihood thresholds
thresholds = [i for i in range(10, 105, 10)]
percentages = {threshold: sum(1 for likelihood in sorted_Projectwise_solving_likelihood.values() if likelihood > threshold) / len(sorted_Projectwise_solving_likelihood) * 100 for threshold in thresholds}

# Print the percentages
for threshold, percentage in percentages.items():
    print(f"Percentage of projects with solving likelihood greater than {threshold}: {percentage:.2f}%")

plt.figure(figsize=(15, 8))
plt.bar(sorted_Projectwise_solving_likelihood.keys(), sorted_Projectwise_solving_likelihood.values(), color='gray')
plt.xlabel('Project')
plt.ylabel('HW Solving Percentage')
plt.title('Project-wise HW Solving Percentage')
plt.xticks(rotation=90)
plt.xlim(-0.5, len(sorted_Projectwise_solving_likelihood) - 0.5)  # Ensures the x-axis starts from 0
plt.ylim(0, max(sorted_Projectwise_solving_likelihood.values(), default=0) * 1.1)  # Ensures the y-axis starts from 0
plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))  # Apply the percentage formatter
plt.tight_layout()
plt.savefig('/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Algorithms/Research Questions/Images/RQ_3_1.png', dpi=1200)
plt.show()