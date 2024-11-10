import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from collections import defaultdict

def count_urls_and_code_snippets(text):
    # Regular expression for detecting URLs
    url_pattern = r'https?://[^\s]+'
    # Regular expression for detecting code snippets (assuming they are enclosed in backticks or triple backticks)
    code_snippet_pattern = r'```[\s\S]*?```|`[^`]*`'

    # Find all matches for URLs and code snippets
    urls = re.findall(url_pattern, text)
    code_snippets = re.findall(code_snippet_pattern, text)

    # Return the counts of URLs and code snippets
    return len(urls), len(code_snippets)

# Function to calculate Mann-Whitney U test and Cliff's delta
def calculate_stats_and_effects(wontfix, non_wontfix):
    stat, p_value = mannwhitneyu(wontfix, non_wontfix)
    delta = cliffs_delta(wontfix, non_wontfix)
    return p_value, delta

# Folder paths containing the CSV files
issue_path = r'F:\RQ Modification\Pre-Solving Likelihood Dataset\Machine Learning Dataset Preparation For All Features Final'

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
HW_tag_came_after_how_many_days = []

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
            if 'help' in str(row['label']).lower() and 'wanted' in str(row['label']).lower():
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
                    count_urls, codesnippets = count_urls_and_code_snippets(str(row['body']))
                    lengthoftheTitle.append(row['lengthofthetitle'])
                    lengthoftheDesc.append(row['lengthofthedesc'])
                    assigned_issue_count.append(row['assigned_issue_count'])
                    latest_assigned_issue_count.append(row['latest_assigned_issue_count'])
                    mentioned_developer.append(row['mentioned_developer'])
                    subscriber.append(row['subscriber'])
                    commentNumber.append(row['commentNumber'])
                    NumberofURls.append(count_urls)
                    NumberofCodeSnippet.append(codesnippets)
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
                    reporter_experience_level_nw.append(repoexplvl==0)
                    reporter_experience_level_md.append(repoexplvl==1)
                    reporter_experience_level_ex.append(repoexplvl==2)
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
    'reporter_experience_level_nw': reporter_experience_level_nw,
    'reporter_experience_level_md': reporter_experience_level_md,
    'reporter_experience_level_ex': reporter_experience_level_ex,
    'HW_tag_came_after_how_many_days': HW_tag_came_after_how_many_days,
    'final_label': final_label
})

# Encode the target variable
data['final_label'] = data['final_label'].apply(lambda x: 1 if x == 'Resolved' else 0)

# Feature selection (excluding the target variable)
features = [
    'length_of_the_Title',
    'length_of_the_Desc',
    'mentioned_developer',
    'subscriber',
    'Total_Comments',
    'Number_of_URls',
    'Number_of_Code_Snippet',
    'expert_comments',
    'mid_level_comments',
    'newcomer_comments',
    'number_of_expert_developers',
    'how_many_expert_developers_committed_recently',
    'number_of_expert_developers_assigned_to_issues',
    'expert_developers_recent_commit_count',
    'recent_PR_resolving_average_time',
    'recent_IR_resolving_average_time',
    'expert_involvement',
    'newcomer_involvement',
    'mid_level_developer_involvement',
    'reporter_experience_level_nw',
    'reporter_experience_level_md',
    'reporter_experience_level_ex',
    'HW_tag_came_after_how_many_days'
]

# Fill missing values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Fit logistic regression model
X = data[features]
y = data['final_label']

X = data.drop(columns=['final_label'])  # Exclude the dependent variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize and fit the XGBoost model
#clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # Set eval_metric to avoid warnings
# Train RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Print accuracy
print("Accuracy on test data: {:.2f}".format(clf.score(X_test, y_test)))

# Compute permutation importance
result = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42)
perm_sorted_idx = result.importances_mean.argsort()

# Plot feature importances
tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.barh(tree_indices, clf.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticks(tree_indices)
ax1.set_yticklabels(X.columns[tree_importance_sorted_idx])  # Use X.columns instead of data.feature_names
ax1.set_ylim((0, len(clf.feature_importances_)))
ax2.boxplot(
    result.importances[perm_sorted_idx].T,
    vert=False,
    labels=X.columns[perm_sorted_idx],  # Use X.columns instead of data.feature_names
)
fig.tight_layout()
plt.show()

# Assuming X is your DataFrame with features
X = X.fillna(X.mean())

# Calculate the correlation matrix
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# Handle potential NaN values in the correlation matrix
corr[np.isnan(corr)] = 0
corr[np.isinf(corr)] = 0

# Compute the distance matrix
distance_matrix = 1 - np.abs(corr)

# Ensure the distance matrix is symmetric
distance_matrix = (distance_matrix + distance_matrix.T) / 2

# Perform hierarchical clustering
dist_linkage = hierarchy.ward(squareform(distance_matrix))

# Plot dendrogram
fig, ax1 = plt.subplots(figsize=(20, 16))
dendro = hierarchy.dendrogram(dist_linkage, labels=X.columns.tolist(), leaf_rotation=90, ax=ax1)
# Adjust the font size of the labels
ax1.axhline(y=0.7, color='blue', linestyle='--', linewidth=1.5)  # Customize color, linestyle, and linewidth as needed


for label in ax1.get_xticklabels():
    label.set_fontsize(18)  # Increase the font size for x-axis labels
    
for label in ax1.get_yticklabels():
    label.set_fontsize(12)  # Increase the font size for y-axis labels


plt.title('Dendrogram', fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.savefig('F:\Research\Final Help Wanted Research Datset and and Code\Algorithms\Research Questions\Images\RQ_3_2.png', dpi=1200)  # Save with 1200 dpi resolution
plt.show()

# Plot heatmap
fig, ax2 = plt.subplots(figsize=(12, 12))
cax = ax2.imshow(corr, cmap='viridis', interpolation='none')
ax2.set_xticks(np.arange(len(X.columns)))
ax2.set_yticks(np.arange(len(X.columns)))
ax2.set_xticklabels(X.columns, rotation=90)
ax2.set_yticklabels(X.columns)

# Add colorbar
cbar = fig.colorbar(cax)
cbar.set_label('Correlation coefficient')

plt.title('Heatmap of Correlation Matrix')
plt.tight_layout()
plt.show()

# Select features from each cluster
cluster_ids = hierarchy.fcluster(dist_linkage, 0.7, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)

# Choose one feature from each cluster
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
print(selected_features)
X_train_sel = X_train.iloc[:, selected_features]
X_test_sel = X_test.iloc[:, selected_features]

selected_feature_names = X.columns[selected_features]

print("Selected feature indices:", selected_features)
print("Selected feature names:", selected_feature_names.tolist())

# Train RandomForestClassifier with selected features
clf_sel = RandomForestClassifier(n_estimators=100, random_state=42)

clf_sel.fit(X_train_sel, y_train)

# Print accuracy of the new model
print(
    "Accuracy on test data with features removed: {:.2f}".format(
        clf_sel.score(X_test_sel, y_test)
    )
)