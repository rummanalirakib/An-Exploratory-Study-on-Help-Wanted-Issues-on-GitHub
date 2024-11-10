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
import textstat
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
    for desc in description:
        has_stack = False
        has_step = False
        has_code = False
        has_patch = False
        has_testcase = False
        has_screenshot = False
        # Check for stack traces
        if re.search(stack_regex, desc):
            has_stack = True

        # Check for patches
        if re.search(patch_regex, desc):
            has_patch = True

        # Check for test cases
        if re.search(testcase_regex, desc):
            has_testcase = True

        # Check for screenshots
        if re.search(screenshot_regex, desc):
            has_screenshot = True

        # Check for steps to reproduce
        for pattern in steps_to_reproduce_patterns:
            if re.search(pattern, desc):
                has_step = True
                break

        # Check for code samples
        if any(construct in desc for construct in code_constructs):
            has_code = True

        completeness_metrics.append([has_stack, has_patch, has_testcase, has_screenshot, has_step, has_code])

    return completeness_metrics

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

def largestConnectedComponen(G, reporterKey):
    # Find connected components
    connected_components = list(nx.connected_components(G))

    # Check if there are connected components
    if not connected_components:
        return False

    # Find the largest connected component
    largest_connected_component = max(connected_components, key=len)

    # Check if the reporterKey is in the largest connected component
    if reporterKey in largest_connected_component:
        return True
    else:
        return False

def ego_betweenness_centrality(G, node):
    ego_network = nx.ego_graph(G, node)
    node_betweenness = 0
    shortest_paths = nx.shortest_path(ego_network, source=node)
    num_shortest_paths = len(shortest_paths)
    for target, path in shortest_paths.items():
        if node != target:  # Exclude paths from node to itself
            num_paths_through_node = sum(1 for p in shortest_paths.values() if node in p and target in p)
            node_betweenness += num_paths_through_node / num_shortest_paths
    return node_betweenness

# Folder path containing the CSV files
folder_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Including Year And All Features Prepared For ML'

all_csv_files = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/HW Issues in csv file'                
# List to store modified DataFrames
modified_dfs = []
filenames = os.listdir(folder_path)
count1 = count2 = 0
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
year_wise = []
business_dimension = []
# Loop through each file in the folder
for filename in filenames:
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(filename)
        mapping = {}
        wontFixTitle = []
        wontFixBody = []
        wontFixLabel = []
        nonWontFixTitle = []
        nonWontFixBody = []
        nonWontFixLabel = []

        wantedBug = {}
        unWantedBug = {}
        wantedRecentBugNum = []
        unwantedRecentBugNum = []
        Bugs = {}
        Reporter = []
        nonReporter = []
        another_file_path = os.path.join(all_csv_files, filename)
        df = pd.read_csv(another_file_path, on_bad_lines='skip', low_memory=False)
        reporter_info = {}
        for index, row in df.iterrows():
            dt = datetime.strptime(row['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            year = dt.year
            reporter_info[row['url']] = year

        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
        InDegree = {}
        OutDegree = {}
        df = df.sort_values(by='url')
        HW_Count = 0
        Othr_Count = 0
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            string_to_label = str(row['label']).lower()

            business_dimension.append([row['openedIRinHW'], row['openedIRinOther'], row['openedPR'], row['numberofexpertdevelopers'],
                                       row['howmanyexpertdeveloperscommittedrecently'], row['numberofdevelopersassignedtoissues'], row['numberofassignedissuestoexpertdevelopers'],
                                       row['expertdevelopersrecentcommitcount'], row['recentPRresolvingaveragetime'], row['recentIRresolvingaveragetime'],
                                       row['number_of_codesnippet'], row['number_of_urls']])
            
            reporterKey = reporter_info[row['url']]
            commentUsers = str(row['prev30DaysComments'])
            title.append(preprocess_text(str(row['title'])))
            body.append(preprocess_text(str(row['body'])))
            
            reporterlvl = 0
            if row['reporter_experience_level'] == 'mid':
                reporterlvl = 1
            elif row['reporter_experience_level'] == 'expert':
                reporterlvl = 2
            
            Reporter_Experience.append([row['bugnum'],  row['latest_bugnum'], row['hw_ratio'], reporterlvl])
            separetedCommentUser = commentUsers.split(',')
            string_to_split = str(row['label']).lower()
            if 'help' in string_to_split or "wanted" in string_to_split:
                 label.append(1)
                 HW_Count += 1
                 count1 += 1
            else:
                 label.append(0)
                 Othr_Count += 1
                 count2 += 1
            Grph = nx.DiGraph()
            
            indegree = {}
            outdegree = {}
            for commentUserKey in separetedCommentUser:
                reporterCommenter = commentUserKey.split(":")
                if len(reporterCommenter) >= 2:
                    Grph.add_edge(reporterCommenter[1], reporterCommenter[0])
                    
                    if reporterCommenter[1] in indegree:
                        indegree[reporterCommenter[1]] += 1
                    else:
                        indegree[reporterCommenter[1]] = 1
                        
                    if reporterCommenter[0] in outdegree:
                        outdegree[reporterCommenter[0]] += 1
                    else:
                        outdegree[reporterCommenter[0]] = 1
                    
                            
            indeg = 0
            outdeg = 0
            
            if reporterKey in indegree:
                indeg = indegree[reporterKey]
            
            if reporterKey in outdegree:
                outdeg = outdegree[reporterKey]
        
            InDegree[reporterKey] = indeg
            OutDegree[reporterKey] = outdeg
        
            total_degree = indeg + outdeg

            NetWorkInDegree.append(indeg)
            NetWorkOutDegree.append(outdeg)
            NetworkTotalDegree.append(total_degree)

            #DensityIn.append(nx.density(Grph.reverse()))
            #DensityOut.append(nx.density(Grph))
            #DensityTotal.append(nx.density(Grph.to_undirected()))
            
            closeness_centrality = nx.closeness_centrality(Grph) 
            betweenness_centrality = nx.betweenness_centrality(Grph)
            clustering_coefficient = nx.clustering(Grph)
            Grph.remove_edges_from(nx.selfloop_edges(Grph))
            k_value = 2
            k_core_subgraph = nx.k_core(Grph, k=k_value)    
            k_coreness = nx.core_number(Grph)

            if reporterKey in Grph:
                ego_betweenness.append(ego_betweenness_centrality(Grph, reporterKey))
            else:
                ego_betweenness.append(0)

           # eigenvector_centrality = nx.eigenvector_centrality(Grph, max_iter=1000, tol=1e-6)

            if reporterKey in betweenness_centrality:
                BetweenNessCentrality.append(betweenness_centrality[reporterKey])
            else:
                BetweenNessCentrality.append(0)

            if reporterKey in clustering_coefficient:
                ClusteringCoefficient.append(clustering_coefficient[reporterKey])
            else:
                ClusteringCoefficient.append(0)

            if reporterKey in k_coreness:
                KCoreNess.append(k_coreness[reporterKey])
            else:
                KCoreNess.append(0)

            if reporterKey in closeness_centrality:
                ClosenessCentrality.append(closeness_centrality[reporterKey])
            else:
                ClosenessCentrality.append(0)

            undirected_graph = Grph.to_undirected()
            LongestConnectedComponent.append(largestConnectedComponen(undirected_graph, reporterKey))
            year_wise.append(reporter_info[row['url']])
            
print(HW_Count, Othr_Count, count1, count2)

df_NetWorkInDegree =  pd.DataFrame(NetWorkInDegree)
df_NetWorkOutDegree = pd.DataFrame(NetWorkOutDegree)
df_NetworkTotalDegree = pd.DataFrame(NetworkTotalDegree)
df_ClosenessCentrality = pd.DataFrame(ClosenessCentrality)
df_LongestConnectedComponent = pd.DataFrame(LongestConnectedComponent)
df_BetweenNessCentrality = pd.DataFrame(BetweenNessCentrality)
#df_EigenVectorCentrality = pd.DataFrame(EigenVectorCentrality)
df_ClusteringCoefficient = pd.DataFrame(ClusteringCoefficient)
df_KCoreNess = pd.DataFrame(KCoreNess)

#df_density_in = pd.DataFrame(DensityIn)
#df_density_out = pd.DataFrame(DensityOut)
#df_density_total = pd.DataFrame(DensityTotal)
df_ego_betweenness = pd.DataFrame(ego_betweenness)

df_NetWorkInDegree =  csr_matrix(df_NetWorkInDegree)
df_NetWorkOutDegree = csr_matrix(df_NetWorkOutDegree)
df_NetworkTotalDegree = csr_matrix(df_NetworkTotalDegree)
df_ClosenessCentrality = csr_matrix(df_ClosenessCentrality)
df_LongestConnectedComponent = csr_matrix(df_LongestConnectedComponent)
df_BetweenNessCentrality = csr_matrix(df_BetweenNessCentrality)
#df_EigenVectorCentrality = csr_matrix(df_EigenVectorCentrality)
df_ClusteringCoefficient = csr_matrix(df_ClusteringCoefficient)
df_KCoreNess = csr_matrix(df_KCoreNess)

#df_density_in = csr_matrix(df_density_in)
#df_density_out = csr_matrix(df_density_out)
#df_density_total = csr_matrix(df_density_total)
df_ego_betweenness = csr_matrix(df_ego_betweenness)
collaboration_network = hstack((df_NetWorkInDegree, df_NetWorkOutDegree, df_NetworkTotalDegree, df_ClosenessCentrality, df_LongestConnectedComponent, 
                                df_BetweenNessCentrality, df_ClusteringCoefficient, df_KCoreNess, df_ego_betweenness))


# Assuming 'title', 'body', and 'label' are lists of titles, bodies, and labels respectively
combined_text = [title[i] + " " + body[i] for i in range(len(title))]

flesch = fog = lix = kincaid = ari = coleman_liau = smog = []

flesch = [textstat.flesch_reading_ease(text) for text in combined_text]
fog = [textstat.gunning_fog(text) for text in combined_text]
lix = [textstat.lix(text) for text in combined_text]
kincaid = [textstat.flesch_kincaid_grade(text) for text in combined_text]
ari = [textstat.automated_readability_index(text) for text in combined_text]
coleman_liau = [textstat.coleman_liau_index(text) for text in combined_text]
smog = [textstat.smog_index(text) for text in combined_text]

flesch_df = pd.DataFrame(flesch)
fog_df = pd.DataFrame(fog)
lix_df = pd.DataFrame(lix)
kincaid_df = pd.DataFrame(kincaid)
ari_df = pd.DataFrame(ari)
coleman_liau_df = pd.DataFrame(coleman_liau)
smog_df = pd.DataFrame(smog)
#merged_df = np.hstack((flesch, fog, lix_df, kincaid_df, ari_df, coleman_liau_df, smog_df))
merged_readability_df = pd.concat([flesch_df, fog_df, lix_df, kincaid_df, ari_df, coleman_liau_df, smog_df], axis=1)

completeness_metrics = extract_technical_information(combined_text)
allMatrics = []

# Convert completeness_metrics to a NumPy array for easier handling
completeness_metrics_array = csr_matrix(completeness_metrics)

Reporter_Experience_array = csr_matrix(Reporter_Experience)

business_dimension_array = csr_matrix(business_dimension)

print('Reporter_Experience_array: ', Reporter_Experience_array.shape)
print('completeness_metrics_array: ', completeness_metrics_array.shape)
print('merged_redability_df: ', merged_readability_df.shape)
print('business_dimension_array: ', business_dimension_array.shape)
merged_redability_df = csr_matrix(merged_readability_df)

vectorizer = CountVectorizer(stop_words='english')
#X_text = vectorizer.fit_transform(combined_text)
tfidf_vectorizer_title = TfidfVectorizer()
X_title = tfidf_vectorizer_title.fit_transform(title)
print(X_title.shape)
idf_threshold = tfidf_vectorizer_title.idf_.max() * 1.0
print("Maximum IDF for title:", idf_threshold)
X_title = X_title[:, tfidf_vectorizer_title.idf_ <= idf_threshold]

# TF-IDF Vectorization for body
tfidf_vectorizer_body = TfidfVectorizer()
X_body = tfidf_vectorizer_body.fit_transform(body)
print(X_body.shape)
idf_threshold = tfidf_vectorizer_body.idf_.max() * 1.0
print("Maximum IDF for body:", idf_threshold)
X_body = X_body[:, tfidf_vectorizer_body.idf_ <= idf_threshold]
X1 = X_title
X2 = X_body
label_encoder = LabelEncoder()
# Encode the target variable
y = label_encoder.fit_transform(label)
y = np.array(y)
X_text = hstack((X1, X2))
# Combine X_text and completeness_metrics_array into a single feature set
# Here, you concatenate the text features with completeness_metrics_array horizontally
year_wise = np.reshape(year_wise, (-1, 1))
label = np.reshape(label, (-1, 1))
print(X1.shape)
print(X2.shape)
print(completeness_metrics_array.shape)
print(merged_redability_df.shape)
print(Reporter_Experience_array.shape)
print(collaboration_network.shape)
print(year_wise.shape)
print(label.shape)

X_combined = hstack((X1, X2, completeness_metrics_array, merged_readability_df, Reporter_Experience_array, collaboration_network, business_dimension_array, label, year_wise))
#X_combined = hstack((X1, X2, completeness_metrics_array, merged_redability_df, Reporter_Experience_array, collaboration_netwrok, business_dimension_array, label))

new_directory_path = r'/home/rumman/Desktop/Research/Final Help Wanted Research Datset and and Code/Local vs Global Dataset'

full_directory_path = os.path.join(new_directory_path, 'Global Dataset1')

# Create the directory if it doesn't exist
os.makedirs(full_directory_path, exist_ok=True)
'''
# Define the paths for the saved files
joblib.dump(X_combined, os.path.join(full_directory_path, 'X_combined.pkl'))
joblib.dump(collaboration_network, os.path.join(full_directory_path, 'collaboration_network.pkl'))
joblib.dump(X_text, os.path.join(full_directory_path, 'X_text.pkl'))
joblib.dump(completeness_metrics_array, os.path.join(full_directory_path, 'completeness_metrics.pkl'))
joblib.dump(Reporter_Experience_array, os.path.join(full_directory_path, 'Reporter_Experience.pkl'))
joblib.dump(business_dimension_array, os.path.join(full_directory_path, 'Business_dimension.pkl'))
joblib.dump(merged_readability_df, os.path.join(full_directory_path, 'Readability.pkl'))
joblib.dump(y, os.path.join(full_directory_path, 'label.pkl'))
joblib.dump(X1, os.path.join(full_directory_path, 'X1.pkl'))
joblib.dump(X2, os.path.join(full_directory_path, 'X2.pkl'))
joblib.dump(year_wise, os.path.join(full_directory_path, 'year_wise.pkl'))
'''