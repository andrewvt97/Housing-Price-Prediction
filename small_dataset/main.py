
from random import randint
import pandas as pd
import numpy as np
import seaborn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
import janitor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
file_name = "NY-House-Dataset.csv"
data = pd.read_csv(file_name)
data = data.clean_names()
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
# BROKER TITLE
# drop this column
# TYPE
# Define categories to group
rare_categories = ['coming soon', 'for sale', 'foreclosure', 'condop for sale', 'mobile house for sale']
merge_to_unknown = lambda x: 'unknown' if x.lower() in rare_categories or x.lower() in ['pending', 'contingent'] else x.lower()
# Apply grouping
train_set['type'] = train_set['type'].apply(merge_to_unknown)
train_set = pd.get_dummies(train_set, columns=['type'], prefix='type')
#PRICE
# BEDS
# keep
# BATHS
# Check for decimal values in the 'bath' column
decimal_bath = train_set['bath'] % 1 != 0
# Drop rows with decimal bath values
train_set = train_set[~decimal_bath] # 243 rows dropped
# Reset the index after dropping rows
train_set.reset_index(drop=True, inplace=True)
# SQUARE FEET
decimal_sqft = train_set['propertysqft'] % 1 != 0
train_set = train_set[~decimal_sqft]
# ADDRESS
# drop
# STATE
# drop
# MAIN ADDRESS
# drop
# ADMINISTRATIVE AREA LEVEL 2
# drop
# LOCALITY
# drop
# SUBLOCALITY
mapping = {
    'Kings County': 'Brooklyn',
    'Brooklyn': 'Brooklyn',
    'New York County': 'Manhattan',
    'Manhattan': 'Manhattan',
    'Bronx County': 'The Bronx',
    'The Bronx': 'The Bronx',
    'Richmond County': 'Staten Island',
    'Staten Island': 'Staten Island',
    'Queens County': 'Queens',
    'Queens': 'Queens',
    'New York': 'Manhattan',
    'Rego Park': 'Queens',
    'Flushing': 'Queens',
    'Riverdale': 'The Bronx',
    'Snyder Avenue': 'Brooklyn',
    'Coney Island': 'Brooklyn',
    'Fort Hamilton': 'Brooklyn'
}
# Apply the mapping to the sublocality column
train_set['sublocality'] = train_set['sublocality'].replace(mapping)
train_set = pd.get_dummies(train_set, columns=['sublocality'], prefix='sublocality')
# train_set = train_set.drop(columns=['sublocality'])
# STREET NAME
# drop
# LONG NAME
# drop
# FORMATTED ADDRESS
# drop
# LATITUDE
# Identify latitude-longitude pairs that have duplicates
duplicates = train_set[train_set.duplicated(subset=['latitude', 'longitude'], keep=False)]
# Remove all rows with duplicate latitude-longitude pairs
train_set = train_set[~train_set[['latitude', 'longitude']].apply(tuple, axis=1).isin(duplicates[['latitude', 'longitude']].apply(tuple, axis=1))]

# Find the optimal number of clusters
coords = train_set[['latitude', 'longitude']]
distortions = []
for k in range(1, 15):  # Test different cluster sizes
    kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
    distortions.append(kmeans.inertia_)
# Apply K-means clustering with the chosen number of clusters 
kmeans = KMeans(n_clusters=6, random_state=42)
train_set['neighborhood_cluster'] = kmeans.fit_predict(train_set[['latitude', 'longitude']])


train_set = train_set.drop(columns=['brokertitle', 'address', 'state', 'main_address', 'administrative_area_level_2',
       'locality',  'street_name', 'long_name',
       'formatted_address']) 
print(train_set.shape)
print(train_set.columns)
correlation_matrix = train_set.corr()
target_correlation = correlation_matrix["price"].sort_values(ascending=False)
target_correlation.to_csv('better_target_correlation.csv')



# For training data
train_features = train_set.drop(columns=['price'])  # Features for training
train_labels = train_set['price']                   # Labels for training


kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Apply log transformation
train_set['log_price'] = np.log1p(train_set['price'])
train_set['propertysqft'] = np.log1p(train_set['propertysqft'])

# Drop missing latitude/longitude rows
train_set = train_set.dropna(subset=['latitude', 'longitude'])

# Prepare features and labels
train_features = train_set.drop(columns=['price', 'log_price'])
train_labels = train_set['log_price']
train_labels_original = np.expm1(train_labels)

# models
# model = XGBRegressor(
#     objective='reg:squarederror',
#     n_estimators=110,       
#     max_depth=8,            
#     learning_rate=0.1,      
#     subsample=0.85,         
#     colsample_bytree=1.0,   
#     random_state=42         
# )
# model = RandomForestRegressor(
#     n_estimators=220,       
#     max_depth=None,         
#     max_features='log2',    
#     min_samples_split=2,    
#     min_samples_leaf=1,    
#     bootstrap=False,        
#     random_state=42,       
#     n_jobs=-1               
# )

# model = MLPRegressor(
#     hidden_layer_sizes=(256, 128, 64),
#     activation='relu',
#     solver='adam',
#     learning_rate_init=0.001,
#     alpha=0.0001,
#     max_iter=500,
#     random_state=42,
#     verbose=True
# )

model = LinearRegression()

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# pred_log = cross_val_predict(model, train_features, train_labels, cv=kf, method='predict')
# pred_original = np.expm1(pred_log)

# # Evaluation
# mape = mean_absolute_percentage_error(train_labels_original, pred_original) * 100
# acc = 100 - mape
# medae = median_absolute_error(train_labels_original, pred_original)
# mae = mean_absolute_error(train_labels_original, pred_original)

# print(f"Train Set Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
# print(f"Train Set Model Accuracy (MAPE): {acc:.2f}%")
# print(f"Train Set Median Absolute Error (MedAE): ${medae:,.2f}")
# print(f"Train Set Mean Absolute Error (MAE): ${mae:,.2f}")


# ### TEST SET PREPROCESSING

# drop columns
test_set = test_set.drop(columns=['brokertitle', 'address', 'state', 'main_address', 'administrative_area_level_2',
       'locality',  'street_name', 'long_name',
       'formatted_address']) 

# TYPE
test_set['type'] = test_set['type'].apply(merge_to_unknown)
test_set = pd.get_dummies(test_set, columns=['type'], prefix='type')

# BATHS
# Check for decimal values in the 'bath' column
decimal_bath = test_set['bath'] % 1 != 0
# Drop rows with decimal bath values
test_set = test_set[~decimal_bath]
# Reset the index after dropping rows
test_set.reset_index(drop=True, inplace=True)
# SQUARE FEET
decimal_sqft = test_set['propertysqft'] % 1 != 0  # Verify rows with decimals
test_set = test_set[~decimal_sqft]
# ADDRESS
# drop
# STATE
# drop
# MAIN ADDRESS
# drop
# ADMINISTRATIVE AREA LEVEL 2
# drop
# LOCALITY
# drop
# SUBLOCALITY
mapping = {
    'Kings County': 'Brooklyn',
    'Brooklyn': 'Brooklyn',
    'New York County': 'Manhattan',
    'Manhattan': 'Manhattan',
    'Bronx County': 'The Bronx',
    'The Bronx': 'The Bronx',
    'Richmond County': 'Staten Island',
    'Staten Island': 'Staten Island',
    'Queens County': 'Queens',
    'Queens': 'Queens',
    'New York': 'Manhattan',
    'Rego Park': 'Queens',
    'Flushing': 'Queens',
    'Riverdale': 'The Bronx',
    'Snyder Avenue': 'Brooklyn',
    'Coney Island': 'Brooklyn',
    'Fort Hamilton': 'Brooklyn'
}
# Apply the mapping to the sublocality column
test_set['sublocality'] = test_set['sublocality'].replace(mapping)
test_set = pd.get_dummies(test_set, columns=['sublocality'], prefix='sublocality')

# STREET NAME
# drop
# LONG NAME
# drop
# FORMATTED ADDRESS
# drop
# LATITUDE

# Identify latitude-longitude pairs that have duplicates
duplicates = test_set[test_set.duplicated(subset=['latitude', 'longitude'], keep=False)]
# Remove all rows with duplicate latitude-longitude pairs
test_set = test_set[~test_set[['latitude', 'longitude']].apply(tuple, axis=1).isin(duplicates[['latitude', 'longitude']].apply(tuple, axis=1))]

# Find the optimal number of clusters
coords = test_set[['latitude', 'longitude']]
distortions = []
for k in range(1, 15):  # Test different cluster sizes
    kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
    distortions.append(kmeans.inertia_)
# Apply K-means clustering 
kmeans = KMeans(n_clusters=6, random_state=42)
test_set['neighborhood_cluster'] = kmeans.fit_predict(test_set[['latitude', 'longitude']])

# Apply log transformation to test set features
test_set['propertysqft'] = np.log1p(test_set['propertysqft'])

# Prepare test features and labels
test_features = test_set.drop(columns=['price'])
test_features = test_features.reindex(columns=train_features.columns, fill_value=0)
test_labels = test_set['price']  # Original scale
log_test_labels = np.log1p(test_labels)  # Log-transformed for model prediction

# Train model on full training data
model.fit(train_features, train_labels)

# Predict log prices and revert to original scale
log_test_preds = model.predict(test_features)
test_preds = np.expm1(log_test_preds)

# Evaluation
mape_test = mean_absolute_percentage_error(test_labels, test_preds) * 100
accuracy_test = 100 - mape_test
medae_test = median_absolute_error(test_labels, test_preds)
mae_test = mean_absolute_error(test_labels, test_preds)

print(f"Test Set Mean Absolute Percentage Error (MAPE): {mape_test:.2f}%")
print(f"Test Set Model Accuracy (MAPE): {accuracy_test:.2f}%")
print(f"Test Set Median Absolute Error (MedAE): ${medae_test:,.2f}")
print(f"Test Set Mean Absolute Error (MAE): ${mae_test:,.2f}")