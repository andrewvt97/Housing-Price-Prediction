
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
import janitor
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
# print(train_set['type'].describe())
# print(train_set['type'].unique())
# print(train_set['type'].value_counts())
# print(train_set['type'].isnull().sum())
# Define categories to group
rare_categories = ['coming soon', 'for sale', 'foreclosure', 'condop for sale', 'mobile house for sale']
merge_to_unknown = lambda x: 'unknown' if x.lower() in rare_categories or x.lower() in ['pending', 'contingent'] else x.lower()
# Apply grouping
train_set['type'] = train_set['type'].apply(merge_to_unknown)
train_set = pd.get_dummies(train_set, columns=['type'], prefix='type')
#PRICE
# print(train_set['price'].describe())
# print(train_set['price'].unique())
# print(train_set['price'].value_counts())
# print(train_set['price'].isnull().sum())
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
print(train_set[decimal_sqft])  # Verify rows with decimals
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
print(train_set['sublocality'].describe())
print(train_set['sublocality'].unique())
print(train_set['sublocality'].value_counts())
print(train_set['sublocality'].isnull().sum())
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
    # Remaining values mapped to Manhattan (assuming "New York" is Manhattan-like)
    'New York': 'Manhattan',
    'Rego Park': 'Queens',
    'Flushing': 'Queens',
    'Riverdale': 'The Bronx',
    'Snyder Avenue': 'Brooklyn',
    'Coney Island': 'Brooklyn',
    'Fort Hamilton': 'Brooklyn'
}
# Apply the mapping to the 'sublocality' column
train_set['sublocality'] = train_set['sublocality'].replace(mapping)
train_set = pd.get_dummies(train_set, columns=['sublocality'], prefix='sublocality')
# drop for now, but promising. Will implement
# train_set = train_set.drop(columns=['sublocality'])
# STREET NAME
# drop
# LONG NAME
# drop
# FORMATTED ADDRESS
# drop
# LATITUDE
print(train_set.shape)
# Identify latitude-longitude pairs that have duplicates
duplicates = train_set[train_set.duplicated(subset=['latitude', 'longitude'], keep=False)]
# Remove all rows with these latitude-longitude pairs
train_set = train_set[~train_set[['latitude', 'longitude']].apply(tuple, axis=1).isin(duplicates[['latitude', 'longitude']].apply(tuple, axis=1))]
# Verify the duplicates have been removed
print("Remaining Duplicate Latitude-Longitude Pairs:")
duplicates = train_set[train_set.duplicated(subset=['latitude', 'longitude'], keep=False)]
print(duplicates)
# Find the optimal number of clusters
coords = train_set[['latitude', 'longitude']]
distortions = []
for k in range(1, 15):  # Test different cluster sizes
    kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
    distortions.append(kmeans.inertia_)
# Apply K-means clustering with the chosen number of clusters (e.g., k=10)
kmeans = KMeans(n_clusters=6, random_state=42)
train_set['neighborhood_cluster'] = kmeans.fit_predict(train_set[['latitude', 'longitude']])

# Confirm updated dataset
train_set = train_set.drop(columns=['brokertitle', 'address', 'state', 'main_address', 'administrative_area_level_2',
       'locality',  'street_name', 'long_name',
       'formatted_address']) 
print(train_set.shape)
print(train_set.columns)
# correlation_matrix = train_set.corr()
# target_correlation = correlation_matrix["price"].sort_values(ascending=False)
# target_correlation.to_csv('better_target_correlation.csv')
# # Create a custom scorer for MAPE (lower is better, so use greater_is_better=False)
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

# For training data
train_features = train_set.drop(columns=['price'])  # Features for training
train_labels = train_set['price']                   # Labels for training
# # Refined parameter grid
# param_grid = {
#     'n_estimators': [90, 100, 110],           # Around the best value of 100
#     'max_depth': [6, 8, 10],                  # Around the best value of 8
#     'learning_rate': [0.09, 0.1, 0.11],       # Narrow range around 0.1067
#     'subsample': [0.75, 0.8, 0.85],           # Around the best value of 0.8
#     'colsample_bytree': [0.9, 1.0]            # Slightly narrow around 1.0
# }
# # Initialize the model with a fixed objective
# model = XGBRegressor(objective='reg:squarederror', random_state=42)
# # # Set up cross-validation strategy
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # Perform grid search
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     scoring=mape_scorer,  # MAPE as the evaluation metric
#     cv=kf,                # Use the same KFold for consistency
#     verbose=2,
#     n_jobs=-1             # Use all available cores
# )
# # Fit the model
# grid_search.fit(features, labels)
# # Print the best parameters and score
# print("Best Parameters (Grid Search):", grid_search.best_params_)
# print("Best MAPE (Grid Search):", -grid_search.best_score_)



kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Final XGBoost model with the best parameters from Grid Search
model = RandomForestRegressor(
    n_estimators=220,       # Optimized number of trees
    max_depth=None,         # Optimized maximum depth (unlimited)
    max_features='log2',    # Optimized feature sampling method
    min_samples_split=2,    # Minimum samples required to split an internal node
    min_samples_leaf=1,     # Minimum samples per leaf node
    bootstrap=False,        # Optimized sampling method
    random_state=42,        # Random state for reproducibility
    n_jobs=-1               # Use all available cores
)
predictions = cross_val_predict(model, train_features, train_labels, cv=kf, method='predict')
mape_original = mean_absolute_percentage_error(train_labels, predictions) * 100
accuracy_original = 100 - mape_original
medae = median_absolute_error(train_labels, predictions)
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(train_labels, predictions))
print(f"Train Set Mean Absolute Percentage Error (MAPE): {mape_original:.2f}%")
print(f"Train Set Model Accuracy (MAPE): {accuracy_original:.2f}%")
print(f"Train Set Median Absolute Error (MedAE): ${medae:,.2f}")
print(f"Train Set Root Mean Squared Error (RMSE): ${rmse:,.2f}")


### TEST SET PREPROCESSING

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
test_set = test_set[~decimal_bath] # 243 rows dropped
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
    # Remaining values mapped to Manhattan (assuming "New York" is Manhattan-like)
    'New York': 'Manhattan',
    'Rego Park': 'Queens',
    'Flushing': 'Queens',
    'Riverdale': 'The Bronx',
    'Snyder Avenue': 'Brooklyn',
    'Coney Island': 'Brooklyn',
    'Fort Hamilton': 'Brooklyn'
}
# Apply the mapping to the 'sublocality' column
test_set['sublocality'] = test_set['sublocality'].replace(mapping)
test_set = pd.get_dummies(test_set, columns=['sublocality'], prefix='sublocality')
# train_set = train_set.drop(columns=['sublocality'])
# STREET NAME
# drop
# LONG NAME
# drop
# FORMATTED ADDRESS
# drop
# LATITUDE

# Identify latitude-longitude pairs that have duplicates
duplicates = test_set[test_set.duplicated(subset=['latitude', 'longitude'], keep=False)]
# Remove all rows with these latitude-longitude pairs
test_set = test_set[~test_set[['latitude', 'longitude']].apply(tuple, axis=1).isin(duplicates[['latitude', 'longitude']].apply(tuple, axis=1))]

# Find the optimal number of clusters
coords = test_set[['latitude', 'longitude']]
distortions = []
for k in range(1, 15):  # Test different cluster sizes
    kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
    distortions.append(kmeans.inertia_)
# Apply K-means clustering with the chosen number of clusters (e.g., k=10)
kmeans = KMeans(n_clusters=6, random_state=42)
test_set['neighborhood_cluster'] = kmeans.fit_predict(test_set[['latitude', 'longitude']])

test_features = test_set.drop(columns=['price']) 
test_features = test_features.reindex(columns=train_features.columns, fill_value=0) 
test_labels = test_set['price']

# Train the model on the full training set
model.fit(train_features, train_labels)
# Generate predictions on the test set
test_predictions = model.predict(test_features)

# Evaluate performance on the test set
mape_test = mean_absolute_percentage_error(test_labels, test_predictions) * 100
accuracy_test = 100 - mape_test
medae_test = median_absolute_error(test_labels, test_predictions)
rmse_test = np.sqrt(mean_squared_error(test_labels, test_predictions))

# Print evaluation metrics
print(f"Test Set Mean Absolute Percentage Error (MAPE): {mape_test:.2f}%")
print(f"Test Set Model Accuracy (MAPE): {accuracy_test:.2f}%")
print(f"Test Set Median Absolute Error (MedAE): ${medae_test:,.2f}")
print(f"Test Set Root Mean Squared Error (RMSE): ${rmse_test:,.2f}")