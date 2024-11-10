import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

new_folder_path = r'D:\Final Help Wanted Research Datset and and Code\HW Issues in csv file'
issueSolverIssueMean = []
issueSolverCommitMean = []
count1 = count2 = count3 = count4 = 0
hw_issues = {}
other_issues = {}

# Loop through each file in the folder
for subdir in os.listdir(new_folder_path):
    print(subdir)
    file_path1 = os.path.join(new_folder_path, subdir)
    try:
        df = pd.read_csv(file_path1, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        continue
    
    for index, row in df.iterrows():
        date_string = str(row['created_at']).strip()
        try:
            if date_string != 'nan' and date_string != '0':  # Checking for 'nan' and '0'
                dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
                year = dt.year
                if 2012 <= year <= 2024:  # Filter years from 2012 to 2024
                    string_to_label = str(row['labels.name']).lower()
                    if 'help' in string_to_label and 'wanted' in string_to_label:
                        if year not in hw_issues:
                            hw_issues[year] = 1
                        else:
                            hw_issues[year] += 1
                    else:
                        if year not in other_issues:
                            other_issues[year] = 1
                        else:
                            other_issues[year] += 1
            else:
                year = None  # Handle case where date_string is 'nan' or '0'
        except ValueError as e:
            print(f"Error parsing date: {e}")
            year = None  # Handle invalid isoformat string

# Calculate the percentage of "Help Wanted" issues year-wise
years = sorted(set(list(hw_issues.keys()) + list(other_issues.keys())))
hw_counts = [hw_issues.get(year, 0) for year in years]
other_counts = [other_issues.get(year, 0) for year in years]

total_counts = [hw + other for hw, other in zip(hw_counts, other_counts)]
hw_percentages = [(hw / total * 100) if total > 0 else 0 for hw, total in zip(hw_counts, total_counts)]

# Filter data from 2012 to 2024
filtered_years = [year for year in years if 2012 <= year <= 2024]
filtered_hw_percentages = [hw_percentages[years.index(year)] for year in filtered_years]

fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axis object

# Plotting Help Wanted Issues percentage
ax.plot(filtered_years, filtered_hw_percentages, marker='o', linestyle='-', color='gray', label='Help Wanted Issues (%)')

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Percentage of Help Wanted Issues')
ax.set_title('Percentage of Help Wanted Issues Over Years (2012-2024)')
ax.legend(loc='upper left')
ax.grid(True)

# Rotate x-axis labels by 45 degrees
ax.tick_params(axis='x', rotation=45)
ax.set_xticks(filtered_years)

plt.tight_layout()

# Save the plot with high resolution
plt.savefig('help_wanted_issues_percentage_plot.png', dpi=1200)  # Adjust dpi as needed for your desired resolution

plt.show()
