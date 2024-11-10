import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Folder paths containing the CSV files
issue_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Machine Learning Dataset Preparation For All Features Final'

# Initialize lists to store data
bugNum = []
latestBugNum = []
reporter_experience_level = []
assigned_issue_count = []
latest_assigned_issue_count = []
mentioned_developer = []
subscriber = []
lengthoftheTitle = []
lengthoftheDesc = []
commentNumber = []
expert_comments = []
mid_level_comments = []
newcomer_comments = []
NumberofURls = []
NumberofCodeSnippet = []
numberofexpertdevelopers = []
howmanyexpertdeveloperscommittedrecently = []
numberofdevelopersassignedtoissues = []
numberofassignedissuestoexpertdevelopers = []
expertdevelopersrecentcommitcount = []
recentPRresolvingaveragetime = []
recentIRresolvingaveragetime = []
expert_involvement = []
newcomer_involvement = []
mid_level_developer_involvement = []
HW_tag_came_after_how_many_days = []
resolution_time = []
issue_solver_commit_count = []
final_label = []

# Load and concatenate the data from CSV files
filenames = sorted(os.listdir(issue_path))

for filename in filenames:
    if filename.endswith('.csv'):
        print(filename)
        # Load resolved issues
        resolved_file_path = os.path.join(issue_path, filename)
        
        df_resolved = pd.read_csv(resolved_file_path, low_memory=False)        
        # Append data to lists
        for index, row in df_resolved.iterrows():
            if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']).lower() and row['resolution time'] != -1:
                repoexplvl = 0
                if row['reporter_experience_level'] == 'mid':
                    repoexplvl = 1
                    
                elif row['reporter_experience_level'] == 'expert':
                    repoexplvl = 2
                    
                expertInvovelment = newcomerInvolvement = midlevelInvolvment = 0
                expertInvovelment = row['expert_comments'] / max(row['expert_comments'] + row['mid_level_comments'] + row['newcomer_comments'] + (repoexplvl==2), 1)
                newcomerInvolvement = row['newcomer_comments'] / max(row['expert_comments'] + row['mid_level_comments'] + row['newcomer_comments'] + (repoexplvl==0), 1)
                midlevelInvolvment = row['mid_level_comments'] / max(row['expert_comments'] + row['mid_level_comments'] + row['newcomer_comments'] + (repoexplvl==2), 1)
                if str(row['issue_solver_commit_count']) != 'nan':
                    lengthoftheTitle.append(row['lengthofthetitle'])
                    lengthoftheDesc.append(row['lengthofthedesc'])
                    assigned_issue_count.append(row['assigned_issue_count'])
                    latest_assigned_issue_count.append(row['latest_assigned_issue_count'])
                    mentioned_developer.append(row['mentioned_developer'])
                    subscriber.append(row['subscriber'])
                    commentNumber.append(row['commentNumber'])
                    NumberofURls.append(row['number_of_urls'])
                    NumberofCodeSnippet.append(row['number_of_codesnippet'])
                    expert_comments.append(row['expert_comments'])
                    mid_level_comments.append(row['mid_level_comments'])
                    newcomer_comments.append(row['newcomer_comments'])
                    numberofexpertdevelopers.append(row['numberofexpertdevelopers'])
                    howmanyexpertdeveloperscommittedrecently.append(row['howmanyexpertdeveloperscommittedrecently'])
                    numberofdevelopersassignedtoissues.append(row['numberofdevelopersassignedtoissues'])
                    numberofassignedissuestoexpertdevelopers.append(row['numberofassignedissuestoexpertdevelopers'])
                    expertdevelopersrecentcommitcount.append(row['expertdevelopersrecentcommitcount'])
                    recentPRresolvingaveragetime.append(row['recentPRresolvingaveragetime'])
                    recentIRresolvingaveragetime.append(row['recentIRresolvingaveragetime'])
                    reporter_experience_level.append(repoexplvl)
                    expert_involvement.append(expertInvovelment)
                    newcomer_involvement.append(newcomerInvolvement)
                    mid_level_developer_involvement.append(midlevelInvolvment)
                    issue_solver_commit_count.append(row['issue_solver_commit_count'])
                    resolution_time.append(row['resolution time'])
                   # HW_tag_came_after_how_many_days.append(row['HW_tag_came_after_how_many_days'])
                    
        

# Create DataFrame from lists
data = pd.DataFrame({
    'length_of_the_Title': lengthoftheTitle,
    'length_of_the_Desc': lengthoftheDesc,
    'mentioned_developer': mentioned_developer,
    'subscriber': subscriber,
    'Total_Comments': commentNumber,
    'Number_of_URls': NumberofURls,
    'Number_of_Code_Snippet': NumberofCodeSnippet,
    'expert_comments': expert_comments,
    'mid_level_comments': mid_level_comments,
    'newcomer_comments': newcomer_comments,
    'number_of_expert_developers': numberofexpertdevelopers,
    'how_many_expert_developers_committed_recently': howmanyexpertdeveloperscommittedrecently,
    'number_of_expert_developers_assigned_to_issues': numberofdevelopersassignedtoissues,
    'expert_developers_recent_commit_count': expertdevelopersrecentcommitcount,
    'recent_PR_resolving_average_time': recentPRresolvingaveragetime,
    'recent_IR_resolving_average_time': recentIRresolvingaveragetime,
    'expert_involvement': expert_involvement,
    'newcomer_involvement': newcomer_involvement,
    'mid_level_developer_involvement': mid_level_developer_involvement,
    'reporter_experience_level': reporter_experience_level,
    'issue_solver_commit_count': issue_solver_commit_count,
  #  'HW_tag_came_after_how_many_days': HW_tag_came_after_how_many_days,
    'resolution_time': resolution_time
})

# Create DataFrame from lists
data = pd.DataFrame({
    'length_of_the_Title': lengthoftheTitle,
    'length_of_the_Desc': lengthoftheDesc,
    'mentioned_developer': mentioned_developer,
    'subscriber': subscriber,
   # 'Total_Comments': commentNumber,
    'Number_of_URls': NumberofURls,
    #'Number_of_Code_Snippet': NumberofCodeSnippet,
    'expert_comments': expert_comments,
    'mid_level_comments': mid_level_comments,
    #'newcomer_comments': newcomer_comments,
    'number_of_expert_developers': numberofexpertdevelopers,
    #'how_many_expert_developers_committed_recently': howmanyexpertdeveloperscommittedrecently,
    #'number_of_expert_developers_assigned_to_issues': numberofdevelopersassignedtoissues,
    #'expert_developers_recent_commit_count': expertdevelopersrecentcommitcount,
    'recent_PR_resolving_average_time': recentPRresolvingaveragetime,
    'recent_IR_resolving_average_time': recentIRresolvingaveragetime,
   # 'expert_involvement': expert_involvement,
    'newcomer_involvement': newcomer_involvement,
    #'mid_level_developer_involvement': mid_level_developer_involvement,
    #'reporter_experience_level': reporter_experience_level,
    'issue_solver_commit_count': issue_solver_commit_count,
  #  'HW_tag_came_after_how_many_days': HW_tag_came_after_how_many_days,
    'resolution_time': resolution_time
})

# Sort by resolution time
data_sorted = data.sort_values(by='resolution_time')

# Calculate the number of rows to label
n_rows = len(data_sorted)
print(n_rows)
top_20_percent = int(n_rows * 0.2)
bottom_20_percent = int(n_rows * 0.2)

# Label the top 20% as 1 and bottom 20% as 0
data_sorted['final_label'] = np.nan  # Initialize with NaN
data_sorted.iloc[:top_20_percent, data_sorted.columns.get_loc('final_label')] = 1
data_sorted.iloc[-bottom_20_percent:, data_sorted.columns.get_loc('final_label')] = 0

# Drop the middle 60%
data_labeled = data_sorted.dropna(subset=['final_label'])

X = data_labeled.drop(columns=['final_label', 'resolution_time'])
y = data_labeled['final_label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize logistic regression model
model = LogisticRegression()

# Fit the model
model.fit(X_scaled, y)

median_values = X.median()

# Create a scenario where 'expert_involvement' varies
scenario_values = median_values.copy()  # Start with median values

probabilities = []

min_expert_involvement = data['subscriber'].min()
max_expert_involvement = data['subscriber'].max()
expert_involvement_values = np.linspace(min_expert_involvement, max_expert_involvement, 1000)

for expert_involvement_value in expert_involvement_values:
    # Set 'expert_involvement' to the current value
    scenario_values['subscriber'] = expert_involvement_value
    # Create a DataFrame with the scenario values
    scenario_data = pd.DataFrame(scenario_values).transpose()
    # Scale the scenario data
    scenario_scaled = scaler.transform(scenario_data)

    # Predict probability for class 1 (resolved issue)
    probability = model.predict_proba(scenario_scaled)[:, 1]
    probabilities.append(probability[0])

min_x_value = min(expert_involvement_values)
min_y_probabilities = min(probabilities)
print(min_y_probabilities)
# Plotting
plt.figure(figsize=(8, 6))
plt.plot(expert_involvement_values, probabilities, marker='o', linestyle='', color='gray')
plt.xlim(min_x_value, max(expert_involvement_values))
plt.ylim(min_y_probabilities, max(probabilities)) 
plt.title('Probability of Issue Resolution Time vs. Subscriber')
plt.xlabel('Subscriber')
plt.ylabel('Probability of Issue Resolution Time')
plt.grid(True)
plt.tight_layout()
plt.show()