import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import networkx as nx
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# List to store modified DataFrames
modified_dfs = []
count1 = 0
title = []
body = []
label = []
keyword_counts = {}
Reporter_Experience = []
NetWorkInDegree = []
NetWorkOutDegree = []
NetworkTotalDegree = []
ClosenessCentrality = []
LongestConnectedComponent = []
BetweenNessCentrality = []
EigenVectorCentrality = []
ClusteringCoefficient = []
KCoreNess = []
DensityIn = []
DensityOut = []
DensityTotal = []
weak_comp_in = []
weak_comp_out = []
weak_comp_total = []
ego_betweenness = []
toJ48 = 0
toRandomForest = 0
toGB = 0
toXB = 0
total = 0
location = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Local vs Global Dataset/Global Dataset'

y = joblib.load(os.path.join(location, 'label.pkl'))
merged_redability_df = joblib.load(os.path.join(location, 'Readability.pkl'))
Reporter_Experience_array = joblib.load(os.path.join(location, 'Reporter_Experience.pkl'))
completeness_metrics_array = joblib.load(os.path.join(location, 'completeness_metrics.pkl'))
X_text = joblib.load(os.path.join(location, 'X_text.pkl'))
X_combined = joblib.load(os.path.join(location, 'X_combined.pkl'))
X1 = joblib.load(os.path.join(location, 'X1.pkl'))
X2 = joblib.load(os.path.join(location, 'X2.pkl'))
collaboration_network = joblib.load(os.path.join(location, 'collaboration_network.pkl'))
Business_dimension = joblib.load(os.path.join(location, 'Business_dimension.pkl'))
year = joblib.load(os.path.join(location, 'year_wise.pkl'))
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_y_pred_randomforresttClassifier = []
all_y_pred_logisticRegression = []
all_y_true = []
all_y_pred_xgboost = []
all_y_pred_j48 = []
all_y_pred_gb = []

X_combined = csr_matrix(X_combined)
# Extract the last column as a sparse matrix
second_last_column = X_combined[:, -1:]
#print(second_last_column)
# Convert last_column to dense format to apply the mask (handling large sparse matrices)
second_last_column_dense = second_last_column.toarray().flatten()
# print(second_last_column_dense)
# Initialize lists to store rows for X_train and X_test
X_train_rows = []
X_test_rows = []

# Initialize lists to store corresponding y values
y_train_rows = []
y_test_rows = []
count1 = count2 = count3 = count4 = count5 = count6  = 0
# Loop over each row to apply the mask
for i in range(second_last_column_dense.shape[0]):
    value = second_last_column_dense[i]
    if 2014 <= value <= 2021:
        X_train_rows.append(X_combined[i, :-2])  # Exclude the last column
        y_train_rows.append(X_combined[i, -2:-1])
        if X_combined[i, -2:-1] == 1:
            count1 += 1
        else:
            count2 += 1
    else:
        X_test_rows.append(X_combined[i, :-2])  # Exclude the last column
        y_test_rows.append(X_combined[i, -2:-1])
        if X_combined[i, -2:-1] == 1:
            count3 += 1
        else:
            count4 += 1

print(count1, count2, count3, count4)
# print(y_train_rows)
# print(y_test_rows)
# Convert lists to sparse matrices
X_train = vstack(X_train_rows)
X_test = vstack(X_test_rows)

print(X_train.shape)
print(X_test.shape)
y_train = np.array(y_train_rows).flatten()
y_train = np.array([x.toarray().flatten() for x in y_train])
y_test = np.array(y_test_rows).flatten()
y_test = np.array([x.toarray().flatten() for x in y_test])
print(y_train.shape, y_test.shape)
business_cal = X_text.shape[1] + completeness_metrics_array.shape[1] + merged_redability_df.shape[1] + Reporter_Experience_array.shape[1] + Business_dimension.shape[1]
reporter_experience_command = X_text.shape[1] + completeness_metrics_array.shape[1] + merged_redability_df.shape[1] + Reporter_Experience_array.shape[1]
readability_Expand = X_text.shape[1] + completeness_metrics_array.shape[1] + merged_redability_df.shape[1]
completenessExpand = X_text.shape[1]+ completeness_metrics_array.shape[1]
X_test_business_dim = X_test[:, business_cal:-2]
X_test_collaboration_network = X_test[:, reporter_experience_command:business_cal]
X_test_reporter_experience = X_test[:, readability_Expand:reporter_experience_command]
X_test_redability = X_test[:, completenessExpand:readability_Expand]
X_test_completeness = X_test[:, X_text.shape[1]:completenessExpand]

X_train_title = X_train[:, :X1.shape[1]]
X_train_description = X_train[:, X1.shape[1]:X1.shape[1]+X2.shape[1]]

X_test_title = X_test[:, :X1.shape[1]]
X_test_description = X_test[:, X1.shape[1]:X1.shape[1]+X2.shape[1]]
print('Test Collaboration Network: ', X_test_collaboration_network.shape)

# Further split the training data into two subsets
X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train, random_state=42)

X_train1_business = X_train1[:, business_cal:-2]
X_train1_collaboration_network = X_train1[:, reporter_experience_command:business_cal]
X_train1_reporter_experience = X_train1[:, readability_Expand:reporter_experience_command]
X_train1_redability = X_train1[:, completenessExpand:readability_Expand]
X_train1_completeness = X_train1[:, X_text.shape[1]:completenessExpand]
X_train1 = X_train1[:, :X_text.shape[1]]

X_train2_business = X_train2[:, business_cal:-2]
X_train2_collaboration_network = X_train2[:, reporter_experience_command:business_cal]
X_train2_reporter_experience = X_train2[:, readability_Expand:reporter_experience_command]
X_train2_redability = X_train2[:, completenessExpand:readability_Expand]
X_train2_completeness = X_train2[:, X_text.shape[1]:completenessExpand]
X_train2 = X_train2[:, :X_text.shape[1]]

X_train1_title = X_train1[:, :X1.shape[1]]
X_train1_description = X_train1[:, X1.shape[1]:X1.shape[1]+X2.shape[1]]

X_train2_title = X_train2[:, :X1.shape[1]]
X_train2_description = X_train2[:, X1.shape[1]:X1.shape[1]+X2.shape[1]]
print(X_train1_title.shape)
print(X_train2_title.shape)
print(y_train1.shape)
print(y_train2.shape)

# Flatten the arrays further to make sure they are 1D
y_train1 = y_train1.flatten()
y_train2 = y_train2.flatten()

print('y_train1 type:', type(y_train1))
print('y_train1 shape:', y_train1.shape)
print('y_train2 type:', type(y_train2))
print('y_train2 shape:', y_train2.shape)
multinomial_nb_classifier_train1_title = MultinomialNB()
multinomial_nb_classifier_train1_title.fit(X_train1_title, y_train1)
multinomial_nb_training_scores_train2_title = multinomial_nb_classifier_train1_title.predict_proba(X_train2_title)[:, 1]

multinomial_nb_classifier_train1_description = MultinomialNB()
multinomial_nb_classifier_train1_description.fit(X_train1_description, y_train1)
multinomial_nb_training_scores_train2_description = multinomial_nb_classifier_train1_description.predict_proba(X_train2_description)[:, 1]

multinomial_nb_classifier_train2_title = MultinomialNB()
multinomial_nb_classifier_train2_title.fit(X_train2_title, y_train2)
multinomial_nb_training_scores_train1_title = multinomial_nb_classifier_train2_title.predict_proba(X_train1_title)[:, 1]

multinomial_nb_classifier_train2_description = MultinomialNB()
multinomial_nb_classifier_train2_description.fit(X_train2_description, y_train2)
multinomial_nb_training_scores_train1_description = multinomial_nb_classifier_train2_description.predict_proba(X_train1_description)[:, 1]

multinomial_nb_training_scores_train1_flat_title = multinomial_nb_training_scores_train1_title.flatten()
multinomial_nb_training_scores_train1_flat_description = multinomial_nb_training_scores_train1_description.flatten()
multinomial_nb_training_scores_train2_flat_title = multinomial_nb_training_scores_train2_title.flatten()
multinomial_nb_training_scores_train2_flat_description = multinomial_nb_training_scores_train2_description.flatten()

# Concatenate the flattened scores
multinomial_nb_combined_scores_title = np.concatenate((multinomial_nb_training_scores_train1_flat_title, multinomial_nb_training_scores_train2_flat_title))
multinomial_nb_combined_scores_description = np.concatenate((multinomial_nb_training_scores_train1_flat_description, multinomial_nb_training_scores_train2_flat_description))


complement_nb_classifier_train1_title  = ComplementNB()
complement_nb_classifier_train1_title.fit(X_train1_title, y_train1)
complement_nb_training_scores_train2_title = complement_nb_classifier_train1_title.predict_proba(X_train2_title)[:, 1]

complement_nb_classifier_train1_description  = ComplementNB()
complement_nb_classifier_train1_description.fit(X_train1_description, y_train1)
complement_nb_training_scores_train2_description = complement_nb_classifier_train1_description.predict_proba(X_train2_description)[:, 1]


complement_nb_classifier_train2_title  = ComplementNB()
complement_nb_classifier_train2_title.fit(X_train2_title, y_train2)
complement_nb_training_scores_train1_title = complement_nb_classifier_train2_title.predict_proba(X_train1_title)[:, 1]

complement_nb_classifier_train2_description  = ComplementNB()
complement_nb_classifier_train2_description.fit(X_train2_description, y_train2)
complement_nb_training_scores_train1_description = complement_nb_classifier_train2_description.predict_proba(X_train1_description)[:, 1]


# Flatten the scores
complement_nb_training_scores_train1_flat_title = complement_nb_training_scores_train1_title.flatten()
complement_nb_training_scores_train2_flat_title= complement_nb_training_scores_train2_title.flatten()
complement_nb_training_scores_train1_flat_description = complement_nb_training_scores_train1_description.flatten()
complement_nb_training_scores_train2_flat_description = complement_nb_training_scores_train2_description.flatten()
# Concatenate the flattened scores
complement_nb_combined_scores_title = np.concatenate((complement_nb_training_scores_train1_flat_title, complement_nb_training_scores_train2_flat_title))
complement_nb_combined_scores_description = np.concatenate((complement_nb_training_scores_train1_flat_description, complement_nb_training_scores_train2_flat_description))

#print(complement_nb_combined_scores)
#print("Size of the matrix:", complement_nb_combined_scores.shape)

discriminative_multinomial_nb_classifier_train1_title = MultinomialNB(alpha=1e-3)
discriminative_multinomial_nb_classifier_train1_title.fit(X_train1_title, y_train1)
discriminative_multinomial_nb_scores_train2_title = discriminative_multinomial_nb_classifier_train1_title.predict_proba(X_train2_title)[:, 1]

discriminative_multinomial_nb_classifier_train1_description = MultinomialNB(alpha=1e-3)
discriminative_multinomial_nb_classifier_train1_description.fit(X_train1_description, y_train1)
discriminative_multinomial_nb_scores_train2_description = discriminative_multinomial_nb_classifier_train1_description.predict_proba(X_train2_description)[:, 1]


discriminative_multinomial_nb_classifier_train2_title = MultinomialNB(alpha=1e-3)
discriminative_multinomial_nb_classifier_train2_title.fit(X_train2_title, y_train2)
discriminative_multinomial_nb_scores_train1_title = discriminative_multinomial_nb_classifier_train2_title.predict_proba(X_train1_title)[:, 1]

discriminative_multinomial_nb_classifier_train2_description = MultinomialNB(alpha=1e-3)
discriminative_multinomial_nb_classifier_train2_description.fit(X_train2_description, y_train2)
discriminative_multinomial_nb_scores_train1_description = discriminative_multinomial_nb_classifier_train2_description.predict_proba(X_train1_description)[:, 1]
# Flatten the scores
discriminative_multinomial_nb_scores_train1_flat_title = discriminative_multinomial_nb_scores_train1_title.flatten()
discriminative_multinomial_nb_scores_train2_flat_title = discriminative_multinomial_nb_scores_train2_title.flatten()
discriminative_multinomial_nb_scores_train1_flat_description = discriminative_multinomial_nb_scores_train1_description.flatten()
discriminative_multinomial_nb_scores_train2_flat_description = discriminative_multinomial_nb_scores_train2_description.flatten()

# Concatenate the flattened scores
discriminative_multinomial_nb_combined_scores_title = np.concatenate((discriminative_multinomial_nb_scores_train1_flat_title, discriminative_multinomial_nb_scores_train2_flat_title))
discriminative_multinomial_nb_combined_scores_description = np.concatenate((discriminative_multinomial_nb_scores_train1_flat_description, discriminative_multinomial_nb_scores_train2_flat_description))

train_completeness_combined = vstack((X_train1_completeness, X_train2_completeness))
train_redability_combined = vstack((X_train1_redability, X_train2_redability))
train_reporter_experience_combined = vstack((X_train1_reporter_experience, X_train2_reporter_experience))
train_collaboration_network = vstack((X_train1_collaboration_network, X_train2_collaboration_network))
X_train_business = vstack((X_train1_business, X_train2_business))

df_multinomial_nb_title = pd.DataFrame(multinomial_nb_combined_scores_title)
df_complement_nb_combined_title = pd.DataFrame(complement_nb_combined_scores_title)
df_discriminative_multinomial_nb_combined_title = pd.DataFrame(discriminative_multinomial_nb_combined_scores_title)
df_multinomial_nb_description = pd.DataFrame(multinomial_nb_combined_scores_description)
df_complement_nb_combined_description = pd.DataFrame(complement_nb_combined_scores_description)
df_discriminative_multinomial_nb_combined_description = pd.DataFrame(discriminative_multinomial_nb_combined_scores_description)

df_multinomial_nb_title = csr_matrix(df_multinomial_nb_title)
df_complement_nb_combined_title = csr_matrix(df_complement_nb_combined_title)
df_discriminative_multinomial_nb_combined_title = csr_matrix(df_discriminative_multinomial_nb_combined_title)
df_multinomial_nb_description = csr_matrix(df_multinomial_nb_description)
df_complement_nb_combined_description = csr_matrix(df_complement_nb_combined_description)
df_discriminative_multinomial_nb_combined_description = csr_matrix(df_discriminative_multinomial_nb_combined_description)
df_train_completeness_combined = csr_matrix(train_completeness_combined)
df_train_redability_combined = csr_matrix(train_redability_combined)
df_train_reporter_experience_combined = csr_matrix(train_reporter_experience_combined)
df_train_collaboration_network = csr_matrix(train_collaboration_network)
df_train_business = csr_matrix(X_train_business)


merged_df = hstack((df_multinomial_nb_title, df_complement_nb_combined_title,
                df_discriminative_multinomial_nb_combined_title, df_multinomial_nb_description,
                df_complement_nb_combined_description, df_discriminative_multinomial_nb_combined_description,
                df_train_completeness_combined, df_train_redability_combined, df_train_reporter_experience_combined, 
                df_train_collaboration_network, df_train_business))
# merged_df = pd.concat([df_multinomial_nb_title, df_complement_nb_combined_title, df_discriminative_multinomial_nb_combined_title, df_nb_combined_title, df_multinomial_nb_description, df_complement_nb_combined_description, df_discriminative_multinomial_nb_combined_description, df_nb_combined_description, df_train_redability_combined, df_train_reporter_experience_combined], axis=1)
print('Merged_DF: ', merged_df.shape)
y_train = np.hstack((y_train1, y_train2))

multinomial_nb_classifier_test_title = MultinomialNB()
multinomial_nb_classifier_test_title.fit(X_train_title, y_train)
multinomial_nb_testing_scores_title = multinomial_nb_classifier_test_title.predict_proba(X_test_title)[:, 1]

multinomial_nb_classifier_test_description = MultinomialNB()
multinomial_nb_classifier_test_description.fit(X_train_description, y_train)
multinomial_nb_testing_scores_description = multinomial_nb_classifier_test_description.predict_proba(X_test_description)[:, 1]

complement_nb_classifier_test_title  = ComplementNB()
complement_nb_classifier_test_title.fit(X_train_title, y_train)
complement_nb_testing_scores_title = complement_nb_classifier_test_title.predict_proba(X_test_title)[:, 1]

complement_nb_classifier_test_description  = ComplementNB()
complement_nb_classifier_test_description.fit(X_train_description, y_train)
complement_nb_testing_scores_description = complement_nb_classifier_test_description.predict_proba(X_test_description)[:, 1]

discriminative_multinomial_nb_classifier_test_title = MultinomialNB(alpha=1e-3)
discriminative_multinomial_nb_classifier_test_title.fit(X_train_title, y_train)
discriminative_multinomial_nb_testing_scores_title = discriminative_multinomial_nb_classifier_test_title.predict_proba(X_test_title)[:, 1]

discriminative_multinomial_nb_classifier_test_description = MultinomialNB(alpha=1e-3)
discriminative_multinomial_nb_classifier_test_description.fit(X_train_description, y_train)
discriminative_multinomial_nb_testing_scores_description = discriminative_multinomial_nb_classifier_test_description.predict_proba(X_test_description)[:, 1]


df_multinomial_nb_test_title = pd.DataFrame(multinomial_nb_testing_scores_title)
df_complement_nb_combined_test_title = pd.DataFrame(complement_nb_testing_scores_title)
df_discriminative_multinomial_nb_combined_test_title = pd.DataFrame(discriminative_multinomial_nb_testing_scores_title)
df_multinomial_nb_test_description = pd.DataFrame(multinomial_nb_testing_scores_description)
df_complement_nb_combined_test_description = pd.DataFrame(complement_nb_testing_scores_description)
df_discriminative_multinomial_nb_combined_test_description = pd.DataFrame(discriminative_multinomial_nb_testing_scores_description)

df_multinomial_nb_test_title = csr_matrix(df_multinomial_nb_test_title)
df_complement_nb_combined_test_title = csr_matrix(df_complement_nb_combined_test_title)
df_discriminative_multinomial_nb_combined_test_title = csr_matrix(df_discriminative_multinomial_nb_combined_test_title)
df_multinomial_nb_test_description = csr_matrix(df_multinomial_nb_test_description)
df_complement_nb_combined_test_description = csr_matrix(df_complement_nb_combined_test_description)
df_discriminative_multinomial_nb_combined_test_description = csr_matrix(df_discriminative_multinomial_nb_combined_test_description)
df_test_completeness = csr_matrix(X_test_completeness)
df_test_redability = csr_matrix(X_test_redability)
df_test_reporter_experience = csr_matrix(X_test_reporter_experience)
df_test_collaboration_network = csr_matrix(X_test_collaboration_network)
df_test_business = csr_matrix(X_test_business_dim)
merged_df_test = hstack((df_multinomial_nb_test_title, df_complement_nb_combined_test_title, df_discriminative_multinomial_nb_combined_test_title,
                        df_multinomial_nb_test_description, df_complement_nb_combined_test_description, df_discriminative_multinomial_nb_combined_test_description,
                        df_test_completeness, df_test_redability, df_test_reporter_experience, df_test_collaboration_network, df_test_business))
#merged_df_test = pd.concat([df_multinomial_nb_test_title, df_complement_nb_combined_test_title, df_discriminative_multinomial_nb_combined_test_title, df_nb_combined_test_title, df_multinomial_nb_test_description, df_complement_nb_combined_test_description, df_discriminative_multinomial_nb_combined_test_description, df_nb_combined_test_description, df_test_redability, df_test_reporter_experience], axis=1)
#smote = SMOTE(random_state=42)
#  merged_df, y_train = smote.fit_resample(merged_df, y_train)
print(merged_df_test.shape)
print(merged_df.shape)
gb_result_append = {}
xgb_result_append = {}
#threshold = 0.4
#all_y_true.extend(y_test)

# Initialize a list to hold the results
results = []

total_positive_class = sum(y_test == 1)
total_negative_class = sum(y_test == 0)
num_top_probabilities = total_positive_class[0]
print(f"Total number of positive class instances in the test dataset: {total_positive_class}", total_negative_class)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(merged_df, y_train)
y_pred_rf = rf_classifier.predict_proba(merged_df_test)    
probs_positive_class_rf = y_pred_rf[:, 1]
probs_and_labels_rf = list(zip(probs_positive_class_rf, y_test))
sorted_probs_and_labels_rf = sorted(probs_and_labels_rf, key=lambda x: x[0], reverse=True)
top_probabilities_rf = sorted_probs_and_labels_rf[:num_top_probabilities]
count_positive_class_rf = sum(1 for prob, label in top_probabilities_rf if label == 1)
print(f"Number of top probabilities from positive class (Random Forest): {count_positive_class_rf}")
results.append(['Random Forest', count_positive_class_rf])

# Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(merged_df, y_train)
y_pred_gb = gb_classifier.predict_proba(merged_df_test)    
probs_positive_class_gb = y_pred_gb[:, 1]
probs_and_labels_gb = list(zip(probs_positive_class_gb, y_test))
sorted_probs_and_labels_gb = sorted(probs_and_labels_gb, key=lambda x: x[0], reverse=True)
top_probabilities_gb = sorted_probs_and_labels_gb[:num_top_probabilities]
count_positive_class_gb = sum(1 for prob, label in top_probabilities_gb if label == 1)
print(f"Number of top probabilities from positive class (Gradient Boosting): {count_positive_class_gb}")
results.append(['Gradient Boosting', count_positive_class_gb])

# XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(random_state=42)
#merged_df_dense = merged_df.astype(float)
xgb_classifier.fit(merged_df, y_train)
y_pred_xgb = xgb_classifier.predict_proba(merged_df_test)    
probs_positive_class_xgb = y_pred_xgb[:, 1]
probs_and_labels_xgb = list(zip(probs_positive_class_xgb, y_test))
sorted_probs_and_labels_xgb = sorted(probs_and_labels_xgb, key=lambda x: x[0], reverse=True)
top_probabilities_xgb = sorted_probs_and_labels_xgb[:num_top_probabilities]
count_positive_class_xgb = sum(1 for prob, label in top_probabilities_xgb if label == 1)
print(f"Number of top probabilities from positive class (XGBoost): {count_positive_class_xgb}")
results.append(['XGBoost', count_positive_class_xgb])

# Average the probabilities
ensemble_probs = (y_pred_rf + y_pred_xgb + y_pred_gb) / 3
probs_and_labels_ensemble = list(zip(ensemble_probs, y_test))
sorted_probs_and_labels_ensemble = sorted(probs_and_labels_ensemble, key=lambda x: x[0], reverse=True)
top_probabilities_ensemble = sorted_probs_and_labels_ensemble[:num_top_probabilities]
count_positive_class_ensemble = sum(1 for prob, label in top_probabilities_ensemble if label == 1)
print(f"Number of top probabilities from positive class (Ensemble): {count_positive_class_ensemble}")