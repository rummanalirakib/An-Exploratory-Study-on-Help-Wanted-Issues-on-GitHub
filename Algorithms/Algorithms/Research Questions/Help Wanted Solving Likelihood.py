import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta
import statsmodels.api as sm
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Function to calculate Mann-Whitney U test and Cliff's delta
def calculate_stats_and_effects(wontfix, non_wontfix):
    stat, p_value = mannwhitneyu(wontfix, non_wontfix)
    delta = cliffs_delta(wontfix, non_wontfix)
    return p_value, delta

# Folder paths containing the CSV files
issue_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Machine Learning Dataset Preparation For All Features Final'

# Initialize lists to store data
bugNum = []
latestBugNum = []
reporter_experience_level_nw = []
reporter_experience_level_md = []
reporter_experience_level_ex = []
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
reporterexperiencelabel = []
final_label = []
HW_tag_came_after_how_many_days = []
count2 = count3 = 0
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
            if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']).lower() and str(row['HW_tag_came_after_how_many_days']) != 'nan':
                repoexplvl = 0
                if row['reporter_experience_level'] == 'mid':
                    repoexplvl = 1
                    
                elif row['reporter_experience_level'] == 'expert':
                    repoexplvl = 2
                    
                expertInvovelment = newcomerInvolvement = midlevelInvolvment = 0
                expertInvovelment = row['expert_comments'] / max(row['expert_comments'] + row['mid_level_comments'] + row['newcomer_comments'] + (repoexplvl==2), 1)
                newcomerInvolvement = row['newcomer_comments'] / max(row['expert_comments'] + row['mid_level_comments'] + row['newcomer_comments'] + (repoexplvl==0), 1)
                midlevelInvolvment = row['mid_level_comments'] / max(row['expert_comments'] + row['mid_level_comments'] + row['newcomer_comments'] + (repoexplvl==2), 1)
                if str(row['issue_solver_commit_count']) != 'nan' or row['state'] == 'open':
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
                  #  reporter_experience_level_nw.append(repoexplvl==0)
                   # reporter_experience_level_md.append(repoexplvl==1)
                   # reporter_experience_level_ex.append(repoexplvl==2)
                    expert_involvement.append(expertInvovelment)
                    newcomer_involvement.append(newcomerInvolvement)
                    mid_level_developer_involvement.append(midlevelInvolvment)
                    HW_tag_came_after_how_many_days.append(row['HW_tag_came_after_how_many_days'])
                    
                if str(row['issue_solver_commit_count']) != 'nan' and row['state'] == 'closed':
                    final_label.append('Resolved')
                elif row['state'] == 'open':
                    final_label.append('Un-resolved')
        

# Create DataFrame from lists
data = pd.DataFrame({
   # 'length_of_the_Title': lengthoftheTitle,
    'length_of_the_Desc': lengthoftheDesc,
    'mentioned_developer': mentioned_developer,
    'subscriber': subscriber,
    #'Total_Comments': commentNumber,
    'Number_of_URls': NumberofURls,
    #'Number_of_Code_Snippet': NumberofCodeSnippet,
   # 'expert_comments': expert_comments,
    #'mid_level_comments': mid_level_comments,
    #'newcomer_comments': newcomer_comments,
    'number_of_expert_developers': numberofexpertdevelopers,
    #'how_many_expert_developers_committed_recently': howmanyexpertdeveloperscommittedrecently,
   # 'number_of_expert_developers_assigned_to_issues': numberofdevelopersassignedtoissues,
    #'expert_developers_recent_commit_count': expertdevelopersrecentcommitcount,
    'recent_PR_resolving_average_time': recentPRresolvingaveragetime,
    #'recent_IR_resolving_average_time': recentIRresolvingaveragetime,
    'expert_involvement': expert_involvement,
    'newcomer_involvement': newcomer_involvement,
    'mid_level_developer_involvement': mid_level_developer_involvement,
   # 'reporter_experience_level_nw': reporter_experience_level_nw,
    'HW_tag_came_after_how_many_days': HW_tag_came_after_how_many_days,
    'final_label': final_label
})

# Encode the target variable
data['final_label'] = data['final_label'].apply(lambda x: 1 if x == 'Resolved' else 0)

# Feature selection (excluding the target variable)
features = [
   # 'length_of_the_Title',
    'length_of_the_Desc',
    'mentioned_developer',
    'subscriber',
   # 'Total_Comments',
    'Number_of_URls',
   # 'Number_of_Code_Snippet',
    #'expert_comments',
    #'mid_level_comments',
    #'newcomer_comments',
    'number_of_expert_developers',
    #'how_many_expert_developers_committed_recently',
   # 'number_of_expert_developers_assigned_to_issues',
    #'expert_developers_recent_commit_count',
    'recent_PR_resolving_average_time',
    #'recent_IR_resolving_average_time',
    'expert_involvement',
    'newcomer_involvement',
    'mid_level_developer_involvement',
    #'reporter_experience_level_nw',
    'HW_tag_came_after_how_many_days'
]


# Fill missing values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Fit logistic regression model
X = data[features]
y = data['final_label']

X = data.drop(columns=['final_label'])  # Exclude the dependent variable
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = sm.Logit(y, sm.add_constant(X))
result = model.fit()

# Print summary of the logistic regression model
print(result.summary())

# Extracting specific values for significant predictors
significant_predictors = result.pvalues[result.pvalues < 0.05].index
for predictor in significant_predictors:
    coef = result.params[predictor]
    std_err = result.bse[predictor]
    z_value = result.tvalues[predictor]
    p_value = result.pvalues[predictor]
    print(f'Predictor: {predictor}')
    print(f'  Coefficient: {coef}')
    print(f'  Standard Error: {std_err}')
    print(f'  z-value: {z_value}')
    print(f'  p-value: {p_value}')
    print()

# Calculate AUC
# Predict probabilities
y_probs = result.predict(sm.add_constant(X))
# Compute AUC score
auc = roc_auc_score(y, y_probs)
print(f'AUC: {auc}')
