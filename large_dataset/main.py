
from random import uniform
import janitor # clean names
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import dask.dataframe as dd
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import gc
file_name = "main_dataset_with_lat_lon.csv"
data = pd.read_csv(file_name)
data = data.clean_names()
# Split data into test and train set
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
# DROP COLUMNS
train_set = train_set.drop(columns=['tax_class_at_present']) # do not care about present tax class
train_set = train_set.drop(columns=['block']) # too many values (19,456) to use in regression model
train_set = train_set.drop(columns=['lot']) # too many values (7,767) inconsequential
train_set = train_set.drop(columns=['building_class_at_present']) # do not care about present building class
train_set = train_set.drop(columns=['address', 'cleaned_address']) # don't care about street name or number (used to get latitude earlier)
train_set = train_set.drop(columns=['apartment_number']) # apartment number has very little value except for floor
train_set = train_set.drop(columns=['_zip_code']) # borough covers the larger areas and neighborhood already covers smaller areas + too many values
# Borough and Neighborhood are dropped later in sale price data cleaning section
# BOROUGH COLUMN
# Convert to numbers and drop rows with null column values
train_set["borough"] = pd.to_numeric(train_set["borough"], errors="coerce")
train_set.dropna(subset=["borough"], inplace=True) # dropped one row

# Keep only rows where borough is 1 and drop all others
# train_set = train_set[train_set["borough"] == 5]
# train_set = pd.get_dummies(train_set, columns=["borough"], prefix="borough", dtype='uint8') # one hot encoding
borough_map = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "StatenIsland"
}
train_set["borough"] = train_set["borough"].map(borough_map)

# NEIGHBORHOOD COLUMN
# Trim white space and standardize the text to uppercase
# train_set['neighborhood'] = train_set['neighborhood'].str.strip().str.upper()
# train_set['neighborhood'] = train_set['neighborhood'].str.replace(r'\s+', ' ', regex=True) # added, recheck later
# # Remove rows where 'neighborhood' contains numeric values or invalid patterns
# train_set = train_set[~train_set['neighborhood'].str.isnumeric()]  # Remove purely numeric neighborhoods
# # Define specific invalid patterns to remove
# invalid_patterns = ['-UNKNOWN', 'NEIGHBORHOOD']
# # # Remove rows where 'neighborhood' contains the invalid patterns (dropped 200 rows)
# train_set = train_set[~train_set['neighborhood'].str.contains('|'.join(invalid_patterns), case=False, na=False)]
# train_set = pd.get_dummies(train_set, columns=['neighborhood'], prefix='neighborhood', dtype='uint8')
# BUILDING CLASS CATEGORY: 
train_set['building_class_category'] = train_set['building_class_category'].str.strip().str.upper()
# Replace multiple spaces with a single space
train_set['building_class_category'] = train_set['building_class_category'].str.replace(r'\s+', ' ', regex=True)
# "01 ONE FAMILY DWELLINGS": "ONE-FAMILY RESIDENTIAL",
#     "01 ONE FAMILY HOMES": "ONE-FAMILY RESIDENTIAL",
#     "02 TWO FAMILY DWELLINGS": "TWO-FAMILY RESIDENTIAL",
#     "02 TWO FAMILY HOMES": "TWO-FAMILY RESIDENTIAL",
#     "03 THREE FAMILY DWELLINGS": "THREE-FAMILY RESIDENTIAL",
#     "03 THREE FAMILY HOMES": "THREE-FAMILY RESIDENTIAL",
category_mapping = {
    "01 ONE FAMILY DWELLINGS": "ONE-FAMILY DWELLINGS",
    "01 ONE FAMILY HOMES": "ONE-FAMILY HOMES",
    "02 TWO FAMILY DWELLINGS": "TWO-FAMILY DWELLINGS",
    "02 TWO FAMILY HOMES": "TWO-FAMILY HOMES",
    "03 THREE FAMILY DWELLINGS": "THREE-FAMILY DWELLINGS",
    "03 THREE FAMILY HOMES": "THREE-FAMILY HOMES",
    "07 RENTALS - WALKUP APARTMENTS": "WALKUP APARTMENTS",
    "12 CONDOS - WALKUP APARTMENTS": "WALKUP APARTMENTS",
    "08 RENTALS - ELEVATOR APARTMENTS": "ELEVATOR APARTMENTS",
    "10 COOPS - ELEVATOR APARTMENTS": "ELEVATOR APARTMENTS",
    "13 CONDOS - ELEVATOR APARTMENTS": "ELEVATOR APARTMENTS",
    "15 CONDOS - 2-10 UNIT RESIDENTIAL": "SMALL CONDOS",
    "16 CONDOS - 2-10 UNIT WITH COMMERCIAL UNIT": "SMALL CONDOS",
    "17 CONDO COOPS": "CONDO COOPS AND CONDOPS",
    "17 CONDOPS": "CONDO COOPS AND CONDOPS",
    "28 COMMERCIAL CONDOS": "COMMERCIAL CONDOS",
    "05 TAX CLASS 1 VACANT LAND": "VACANT LAND (RESIDENTIAL)",
    "31 COMMERCIAL VACANT LAND": "VACANT LAND (COMMERCIAL)",
    "21 OFFICE BUILDINGS": "OFFICE BUILDINGS",
    "22 STORE BUILDINGS": "STORE BUILDINGS",
    "43 CONDO OFFICE BUILDINGS": "CONDO OFFICE SPACES",
    "29 COMMERCIAL GARAGES": "PARKING AND STORAGE",
    "44 CONDO PARKING": "PARKING AND STORAGE",
    "47 CONDO NON-BUSINESS STORAGE": "PARKING AND STORAGE",
    "11 SPECIAL CONDO BILLING LOTS": "PARKING AND STORAGE",
    "27 FACTORIES": "INDUSTRIAL",
    "30 WAREHOUSES": "INDUSTRIAL",
    "49 CONDO WAREHOUSES/FACTORY/INDUS": "INDUSTRIAL",
    "37 RELIGIOUS FACILITIES": "RELIGIOUS",
    "32 HOSPITAL AND HEALTH FACILITIES": "HEALTH AND HOSPITALS",
    "33 EDUCATIONAL FACILITIES": "EDUCATIONAL FACILITIES",
    "35 INDOOR PUBLIC AND CULTURAL FACILITIES": "CULTURAL CENTERS",
    "42 CONDO CULTURAL/MEDICAL/EDUCATIONAL/ETC": "CULTURAL CENTERS",
    "36 OUTDOOR RECREATIONAL FACILITIES": "OUTDOOR RECREATION",
    "34 THEATRES": "THEATRES",
    "18 TAX CLASS 3 - UNTILITY PROPERTIES": "UTILITIES",
    "18 TAX CLASS 3 - UNTILITY PROPERTIES": "UTILITIES",
    "39 TRANSPORTATION FACILITIES": "TRANSPORTATION FACILITIES",
    "40 SELECTED GOVERNMENTAL FACILITIES": "GOVERNMENT PROPERTIES",
    "24 TAX CLASS 4 - UTILITY BUREAU PROPERTIES": "GOVERNMENT PROPERTIES",
    "25 LUXURY HOTELS": "LUXURY HOTELS",
    "26 OTHER HOTELS": "OTHER HOTELS",
    "45 CONDO HOTELS": "CONDO HOTELS",
}
# Map the categories
train_set['building_class_category'] = train_set['building_class_category'].replace(category_mapping)
# # Drop rows where 'building_class_category' is not in the mapping or invalid
valid_categories = list(category_mapping.values())
train_set = train_set[train_set['building_class_category'].isin(valid_categories)] # (estimated 100,000 rows dropped)
# Trying to train only houses (57,000 rows dropped)
# house_categories = [
#     "ONE-FAMILY RESIDENTIAL",
#     # "TWO-FAMILY RESIDENTIAL",
#     # "THREE-FAMILY RESIDENTIAL",
# ]
house_categories = [
    "ONE-FAMILY DWELLINGS",
    "ONE-FAMILY HOMES", 
    # "TWO-FAMILY DWELLINGS",
    # "TWO-FAMILY HOMES", #
    # "THREE-FAMILY DWELLINGS", 
    # "THREE-FAMILY HOMES", 
]
# Filter dataset to keep only single family houses
train_set = train_set[train_set['building_class_category'].isin(house_categories)]
# One hot encoding for building class category is done in the land square feet section
# EASEMENT:
# Replace 'E' and 'EASEMENT' with 1, and everything else with 0
train_set['ease_ment'] = train_set['ease_ment'].replace({'E': 1, 'EASEMENT': 1}, regex=False)
train_set['ease_ment'] = train_set['ease_ment'].replace(' ', 0, regex=False)
train_set = train_set.assign(ease_ment=train_set['ease_ment'].fillna(0).astype(int))
# RESIDENTIAL UNITS
train_set = train_set[~train_set['residential_units'].isin(['RESIDENTIAL UNITS'])]
train_set['residential_units'] = pd.to_numeric(train_set['residential_units'], errors='coerce')
train_set.dropna(subset=['residential_units'], inplace=True) # drops 100000 rows
train_set['residential_units'] = train_set['residential_units'].astype(int)
# COMMERCIAL UNITS
train_set = train_set[~train_set['commercial_units'].isin(['COMMERCIAL UNITS'])]  # Remove invalid rows
train_set['commercial_units'] = pd.to_numeric(train_set['commercial_units'], errors='coerce')  # Convert to numeric
train_set['commercial_units'] = train_set['commercial_units'].fillna(0)
train_set['commercial_units'] = train_set['commercial_units'].astype(int)  # Convert to integers
# TOTAL UNITS:
#handle invalid entries
train_set = train_set[~train_set['total_units'].isin(['TOTAL UNITS'])]  # Remove invalid rows
train_set['total_units'] = pd.to_numeric(train_set['total_units'], errors='coerce')  # Convert to numeric
train_set.dropna(subset=['total_units'], inplace=True)  # Drop rows with NaN values (9000 affected) - 0 affected after previous drops
train_set['total_units'] = train_set['total_units'].astype(int)  # Convert to integers
# LAND SQUARE FEET:
condo_types = ['ELEVATOR APARTMENTS', 'WALKUP APARTMENTS', 'SMALL CONDOS', 'CONDO COOPS AND CONDOPS']
train_set.loc[train_set['building_class_category'].isin(condo_types), 'land_square_feet'] = train_set.loc[
    train_set['building_class_category'].isin(condo_types), 'land_square_feet'].fillna(0)
train_set['land_square_feet'] = pd.to_numeric(train_set['land_square_feet'], errors='coerce')
train_set.dropna(subset=['land_square_feet'], inplace=True) # drops 1,000 rows if condos are assigned 0 - around 300 after previous drops
train_set = pd.get_dummies(train_set, columns=['building_class_category'], prefix='class', dtype='uint8') # does one hot encoding for building class category
train_set = train_set[train_set['land_square_feet'] >= 50] # Remove rows where land_square_feet is less than 50 (should be 0 when including apartments)
# GROSS SQUARE FEET:
train_set['gross_square_feet'] = pd.to_numeric(train_set['gross_square_feet'], errors='coerce')
train_set.dropna(subset=['gross_square_feet'], inplace=True) # drops around 50000 rows
train_set = train_set[train_set['gross_square_feet'] >= 50]  # Remove rows where gross_square_feet is less than 50

# TAX CLASS AT TIME OF SALE
train_set['tax_class_at_time_of_sale'] = pd.to_numeric(train_set['tax_class_at_time_of_sale'], errors='coerce').astype('uint8')
train_set.dropna(subset=['tax_class_at_time_of_sale'], inplace=True) # nothing dropped
train_set = pd.get_dummies(train_set, columns=['tax_class_at_time_of_sale'], prefix='tax_class', dtype='uint8')
# BUILDING CLASS AT TIME OF SALE
train_set = pd.get_dummies(train_set, columns=['building_class_at_time_of_sale'], prefix='building_class', drop_first=True, dtype='uint8')
# LAT and LONG
train_set = train_set.dropna(subset=['latitude', 'longitude'])#2384
# SALE DATE
# Load the quarterly house price index data
hpi_data = pd.read_csv("NYSTHPI.csv")
hpi_data['observation_date'] = pd.to_datetime(hpi_data['observation_date'])  # Ensure correct datetime format
hpi_data['year'] = hpi_data['observation_date'].dt.year
# Infer the quarter from the observation_date
hpi_data['quarter'] = (hpi_data['observation_date'].dt.month - 1) // 3 + 1  # Map months to quarters
# Ensure sale_date is in the correct datetime format and drop nulls
train_set['sale_date'] = pd.to_datetime(train_set['sale_date'], errors='coerce')  # Convert to datetime, coerce invalid entries to NaT
train_set.dropna(subset=['sale_date'], inplace=True)  # Drop rows where sale_date is NaT
# Extract year and quarter from sale_date
train_set['sale_year'] = train_set['sale_date'].dt.year
train_set['sale_quarter'] = train_set['sale_date'].dt.quarter
# Merge train_set with HPI data based on year and quarter
train_set = pd.merge(
    train_set,
    hpi_data[['year', 'quarter', 'NYSTHPI']],
    left_on=['sale_year', 'sale_quarter'],
    right_on=['year', 'quarter'],
    how='left'
)
# Rename the merged HPI column
train_set.rename(columns={'NYSTHPI': 'quarterly_price_index'}, inplace=True)
# Drop unnecessary columns
train_set.drop(columns=['sale_quarter', 'year', 'quarter'], inplace=True)
# YEAR BUILT
train_set['year_built'] = pd.to_numeric(train_set['year_built'], errors='coerce')
train_set.dropna(subset=['year_built'], inplace=True)
train_set['year_built'] = train_set['year_built'].astype(int)
train_set = train_set[(train_set['year_built'] >= 1800) & (train_set['year_built'] <= 2024)]
# Calculate house_age at sale
train_set['house_age'] = train_set['sale_year'] - train_set['year_built']
train_set = train_set[train_set['house_age'] >= 0]
train_set.drop(columns=['sale_date', 'sale_year'], inplace=True)
train_set.drop(columns=['year_built'], inplace=True)
# SALE PRICE
# Remove any values with Sale Price that fall in Gift Range or invalid values
train_set["sale_price"] = pd.to_numeric(train_set["sale_price"], errors="coerce")
train_set.dropna(subset=["sale_price"], inplace=True)
# Define borough-specific lower bound rates
borough_lower_bounds = {
    "StatenIsland": 82,
    "Brooklyn": 137,
    "Bronx": 229,
    "Queens": 229,
    "Manhattan": 1740.3
}
# Apply borough-specific lower bound filtering
train_set = train_set[
    train_set.apply(lambda row: row['sale_price'] >= row['quarterly_price_index'] * borough_lower_bounds.get(row['borough'], 82), axis=1)
]

# Apply upper bound filtering (remains the same for all boroughs)
upper_bound = train_set['quarterly_price_index'] * 138000
train_set = train_set[train_set['sale_price'] <= upper_bound]
# DROP LOCATIONS because latitude and longitude was added
train_set = train_set.drop(columns=['neighborhood','borough'])

# REMOVING OUTLIERS (Not needed for residential single family homes)
# Residential Units
# Calculate the 99th percentile
# upper_limit = train_set['residential_units'].quantile(0.99)
# train_set['residential_units'] = train_set['residential_units'].clip(upper=upper_limit)
# Commercial Units (No effect on residential housing - keep commented)
# # Calculate the 99th percentile
# upper_limit = train_set['commercial_units'].quantile(0.99)
# train_set['commercial_units'] = train_set['commercial_units'].clip(upper=upper_limit)
# Total Units
# # Calculate the 99th percentile
# upper_limit = train_set['total_units'].quantile(0.99)
# train_set['total_units'] = train_set['total_units'].clip(upper=upper_limit)
# Land Square Feet
# Cap at the 99th percentile to reduce the effect of outliers
# upper_limit = train_set['land_square_feet'].quantile(0.99)
# train_set['land_square_feet'] = train_set['land_square_feet'].clip(upper=upper_limit)
# Gross Square Feet
# Cap at the 99th percentile to reduce the effect of outliers
# upper_limit = train_set['gross_square_feet'].quantile(0.99)
# train_set['gross_square_feet'] = train_set['gross_square_feet'].clip(upper=upper_limit)


# RESIDENTIAL SINGLE FAMILY ONLY SPECIFIC CLEANING
# Apply filters to keep only single-family residential homes with no commercial units
train_set = train_set[
    (train_set['residential_units'] == 1) &
    (train_set['total_units'] == 1) &
    (train_set['commercial_units'] == 0)
]
train_set = train_set[train_set['land_square_feet'] >= train_set['gross_square_feet']] 

# CALCULATE CORRELATION OF FEATURES
correlation_matrix = train_set.corr()
target_correlation = correlation_matrix["sale_price"].sort_values(ascending=False)
target_correlation.to_csv('target_correlation.csv')

# target_correlation = pd.read_csv('target_correlation.csv', index_col=0)
#  # Convert to a Series
# if isinstance(target_correlation, pd.DataFrame):
#     target_correlation = target_correlation.iloc[:, 0]
# # Replace NaN or missing values with 0 
# target_correlation = target_correlation.fillna(0)
# # Define a correlation threshold
# correlation_threshold = 0.01 # 0.01
# # Filter out columns based on the threshold
# columns_to_keep = target_correlation[target_correlation.abs() >= correlation_threshold].index
# columns_to_drop = target_correlation[target_correlation.abs() < correlation_threshold].index

# Drop columns below the threshold from dataset
# train_set = train_set.drop(columns=columns_to_drop)

# APPLY LOG TRANSFORMATIONS
train_set['log_sale_price'] = np.log1p(train_set['sale_price'])
train_set['residential_units'] = np.log1p(train_set['residential_units'])
train_set['commercial_units'] = np.log1p(train_set['commercial_units'])
train_set['total_units'] = np.log1p(train_set['total_units'])
train_set['land_square_feet'] = np.log1p(train_set['land_square_feet'])
train_set['gross_square_feet'] = np.log1p(train_set['gross_square_feet'])
# Years Since Sale (Log Transformation Defintely Not Needed)
# House Age (Log Transformation Likely Not Needed)
print(train_set.shape)


# # Prepare features and labels
train_features = train_set.drop(columns=['sale_price', 'log_sale_price'])  # Drop both original and log-transformed sale prices
train_labels = train_set['log_sale_price']  # Use the log-transformed target
train_labels_original = np.expm1(train_labels)  # Convert log-transformed labels back to original sale prices
# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # parameters for XGBoost
# model = XGBRegressor(
#     objective='reg:squarederror',
#     n_estimators=110,       
#     max_depth=8,            
#     learning_rate=0.1,     
#     subsample=0.85,         
#     colsample_bytree=1.0,   
#     random_state=41         
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

model = LinearRegression()
# model = MLPRegressor(
#     hidden_layer_sizes=(256, 128, 64),
#     activation='relu',
#     solver='adam',
#     learning_rate_init=0.001,
#     alpha=0.0001,
#     max_iter=500,
#     early_stopping=True,
#     n_iter_no_change=10,
#     random_state=42,
#     verbose=False
# )

# # Perform K-Fold cross-validation
train_predictions_log = cross_val_predict(model, train_features, train_labels, cv=kf, method='predict')
train_predictions_original = np.expm1(train_predictions_log)  # Convert predictions back to the original scale


# Calculate metrics for the model
train_mape_original = mean_absolute_percentage_error(train_labels_original, train_predictions_original) * 100
train_accuracy_original = 100 - train_mape_original
train_medae = median_absolute_error(train_labels_original, train_predictions_original)
train_mae = mean_absolute_error(train_labels_original, train_predictions_original)
print(f"Train Set Mean Absolute Percentage Error (MAPE): {train_mape_original:.2f}%")
print(f"Train Set Model Accuracy (MAPE): {train_accuracy_original:.2f}%")
print(f"Train Set Median Absolute Error (MedAE): ${train_medae:,.2f}")
print(f"Train Set Mean Absolute Error (MAE): ${train_mae:,.2f}")



# ########################
#  TESTING
# #######################

# DROP COLUMNS
test_set = test_set.drop(columns=['tax_class_at_present']) 
test_set = test_set.drop(columns=['block']) 
test_set = test_set.drop(columns=['lot']) #
test_set = test_set.drop(columns=['building_class_at_present']) 
test_set = test_set.drop(columns=['address', 'cleaned_address']) 
test_set = test_set.drop(columns=['apartment_number']) 
test_set = test_set.drop(columns=['_zip_code']) 

# BOROUGH COLUMN
# Convert to numbers and drop rows with null column values
test_set["borough"] = pd.to_numeric(test_set["borough"], errors="coerce")
test_set.dropna(subset=["borough"], inplace=True) 
# Keep only rows where borough is 1 and drop all others
# test_set = test_set[test_set["borough"] == 5]
# test_set = pd.get_dummies(test_set, columns=["borough"], prefix="borough", dtype='uint8') # one hot encoding
test_set["borough"] = test_set["borough"].map(borough_map)

# NEIGHBORHOOD COLUMN
# Trim white space and standardize the text to uppercase
# test_set['neighborhood'] = test_set['neighborhood'].str.strip().str.upper()
# test_set['neighborhood'] = test_set['neighborhood'].str.replace(r'\s+', ' ', regex=True) 
# # Remove rows where 'neighborhood' contains numeric values or invalid patterns
# test_set = test_set[~test_set['neighborhood'].str.isnumeric()]  # Remove purely numeric neighborhoods
# # Define specific invalid patterns to remove
# invalid_patterns = ['-UNKNOWN', 'NEIGHBORHOOD']
# # # Remove rows where 'neighborhood' contains the invalid patterns 
# test_set = test_set[~test_set['neighborhood'].str.contains('|'.join(invalid_patterns), case=False, na=False)]
# test_set = pd.get_dummies(test_set, columns=['neighborhood'], prefix='neighborhood', dtype='uint8')
# BUILDING CLASS CATEGORY: 
test_set['building_class_category'] = test_set['building_class_category'].str.strip().str.upper()
# Replace multiple spaces with a single space
test_set['building_class_category'] = test_set['building_class_category'].str.replace(r'\s+', ' ', regex=True)
# "01 ONE FAMILY DWELLINGS": "ONE-FAMILY RESIDENTIAL",
#     "01 ONE FAMILY HOMES": "ONE-FAMILY RESIDENTIAL",
#     "02 TWO FAMILY DWELLINGS": "TWO-FAMILY RESIDENTIAL",
#     "02 TWO FAMILY HOMES": "TWO-FAMILY RESIDENTIAL",
#     "03 THREE FAMILY DWELLINGS": "THREE-FAMILY RESIDENTIAL",
#     "03 THREE FAMILY HOMES": "THREE-FAMILY RESIDENTIAL",
category_mapping = {
    "01 ONE FAMILY DWELLINGS": "ONE-FAMILY DWELLINGS",
    "01 ONE FAMILY HOMES": "ONE-FAMILY HOMES",
    "02 TWO FAMILY DWELLINGS": "TWO-FAMILY DWELLINGS",
    "02 TWO FAMILY HOMES": "TWO-FAMILY HOMES",
    "03 THREE FAMILY DWELLINGS": "THREE-FAMILY DWELLINGS",
    "03 THREE FAMILY HOMES": "THREE-FAMILY HOMES",
    "07 RENTALS - WALKUP APARTMENTS": "WALKUP APARTMENTS",
    "12 CONDOS - WALKUP APARTMENTS": "WALKUP APARTMENTS",
    "08 RENTALS - ELEVATOR APARTMENTS": "ELEVATOR APARTMENTS",
    "10 COOPS - ELEVATOR APARTMENTS": "ELEVATOR APARTMENTS",
    "13 CONDOS - ELEVATOR APARTMENTS": "ELEVATOR APARTMENTS",
    "15 CONDOS - 2-10 UNIT RESIDENTIAL": "SMALL CONDOS",
    "16 CONDOS - 2-10 UNIT WITH COMMERCIAL UNIT": "SMALL CONDOS",
    "17 CONDO COOPS": "CONDO COOPS AND CONDOPS",
    "17 CONDOPS": "CONDO COOPS AND CONDOPS",
    "28 COMMERCIAL CONDOS": "COMMERCIAL CONDOS",
    "05 TAX CLASS 1 VACANT LAND": "VACANT LAND (RESIDENTIAL)",
    "31 COMMERCIAL VACANT LAND": "VACANT LAND (COMMERCIAL)",
    "21 OFFICE BUILDINGS": "OFFICE BUILDINGS",
    "22 STORE BUILDINGS": "STORE BUILDINGS",
    "43 CONDO OFFICE BUILDINGS": "CONDO OFFICE SPACES",
    "29 COMMERCIAL GARAGES": "PARKING AND STORAGE",
    "44 CONDO PARKING": "PARKING AND STORAGE",
    "47 CONDO NON-BUSINESS STORAGE": "PARKING AND STORAGE",
    "11 SPECIAL CONDO BILLING LOTS": "PARKING AND STORAGE",
    "27 FACTORIES": "INDUSTRIAL",
    "30 WAREHOUSES": "INDUSTRIAL",
    "49 CONDO WAREHOUSES/FACTORY/INDUS": "INDUSTRIAL",
    "37 RELIGIOUS FACILITIES": "RELIGIOUS",
    "32 HOSPITAL AND HEALTH FACILITIES": "HEALTH AND HOSPITALS",
    "33 EDUCATIONAL FACILITIES": "EDUCATIONAL FACILITIES",
    "35 INDOOR PUBLIC AND CULTURAL FACILITIES": "CULTURAL CENTERS",
    "42 CONDO CULTURAL/MEDICAL/EDUCATIONAL/ETC": "CULTURAL CENTERS",
    "36 OUTDOOR RECREATIONAL FACILITIES": "OUTDOOR RECREATION",
    "34 THEATRES": "THEATRES",
    "18 TAX CLASS 3 - UNTILITY PROPERTIES": "UTILITIES",
    "18 TAX CLASS 3 - UNTILITY PROPERTIES": "UTILITIES",
    "39 TRANSPORTATION FACILITIES": "TRANSPORTATION FACILITIES",
    "40 SELECTED GOVERNMENTAL FACILITIES": "GOVERNMENT PROPERTIES",
    "24 TAX CLASS 4 - UTILITY BUREAU PROPERTIES": "GOVERNMENT PROPERTIES",
    "25 LUXURY HOTELS": "LUXURY HOTELS",
    "26 OTHER HOTELS": "OTHER HOTELS",
    "45 CONDO HOTELS": "CONDO HOTELS",
}
# Map the categories
test_set['building_class_category'] = test_set['building_class_category'].replace(category_mapping)
# # Drop rows where 'building_class_category' is not in the mapping or invalid
valid_categories = list(category_mapping.values())
test_set = test_set[test_set['building_class_category'].isin(valid_categories)] 

# house_categories = [
#     "ONE-FAMILY RESIDENTIAL",
#     # "TWO-FAMILY RESIDENTIAL",
#     # "THREE-FAMILY RESIDENTIAL",
# ]
house_categories = [
    "ONE-FAMILY DWELLINGS",
    "ONE-FAMILY HOMES", 
    # "TWO-FAMILY DWELLINGS",
    # "TWO-FAMILY HOMES", 
    # "THREE-FAMILY DWELLINGS",
    # "THREE-FAMILY HOMES",
]
# Filter dataset to keep only single family houses
test_set = test_set[test_set['building_class_category'].isin(house_categories)]
# One hot encoding for building class category is done in the land square feet section
# EASEMENT:
# Replace 'E' and 'EASEMENT' with 1, and everything else with 0
test_set['ease_ment'] = test_set['ease_ment'].replace({'E': 1, 'EASEMENT': 1}, regex=False)
test_set['ease_ment'] = test_set['ease_ment'].replace(' ', 0, regex=False)
test_set = test_set.assign(ease_ment=test_set['ease_ment'].fillna(0).astype(int))
# RESIDENTIAL UNITS
test_set = test_set[~test_set['residential_units'].isin(['RESIDENTIAL UNITS'])]
test_set['residential_units'] = pd.to_numeric(test_set['residential_units'], errors='coerce')
test_set.dropna(subset=['residential_units'], inplace=True) 
test_set['residential_units'] = test_set['residential_units'].astype(int)
# COMMERCIAL UNITS
test_set = test_set[~test_set['commercial_units'].isin(['COMMERCIAL UNITS'])]  # Remove invalid rows
test_set['commercial_units'] = pd.to_numeric(test_set['commercial_units'], errors='coerce')  # Convert to numeric
test_set['commercial_units'] = test_set['commercial_units'].fillna(0)
test_set['commercial_units'] = test_set['commercial_units'].astype(int)  # Convert to integers
# TOTAL UNITS:
# handle invalid entries
test_set = test_set[~test_set['total_units'].isin(['TOTAL UNITS'])]  # Remove invalid rows
test_set['total_units'] = pd.to_numeric(test_set['total_units'], errors='coerce')  # Convert to numeric
test_set.dropna(subset=['total_units'], inplace=True)  
test_set['total_units'] = test_set['total_units'].astype(int)  # Convert to integers
# LAND SQUARE FEET:
condo_types = ['ELEVATOR APARTMENTS', 'WALKUP APARTMENTS', 'SMALL CONDOS', 'CONDO COOPS AND CONDOPS']
test_set.loc[test_set['building_class_category'].isin(condo_types), 'land_square_feet'] = test_set.loc[
    test_set['building_class_category'].isin(condo_types), 'land_square_feet'].fillna(0)
test_set['land_square_feet'] = pd.to_numeric(test_set['land_square_feet'], errors='coerce')
test_set.dropna(subset=['land_square_feet'], inplace=True) 
test_set = pd.get_dummies(test_set, columns=['building_class_category'], prefix='class', dtype='uint8') # does one hot encoding for building class category
test_set = test_set[test_set['land_square_feet'] >= 50] # Remove rows where land_square_feet is less than 50 (should be 0 if apartments included)
# GROSS SQUARE FEET:
test_set['gross_square_feet'] = pd.to_numeric(test_set['gross_square_feet'], errors='coerce')
test_set.dropna(subset=['gross_square_feet'], inplace=True)
test_set = test_set[test_set['gross_square_feet'] >= 50]  # Remove rows where gross_square_feet is less than 50

# TAX CLASS AT TIME OF SALE
test_set['tax_class_at_time_of_sale'] = pd.to_numeric(test_set['tax_class_at_time_of_sale'], errors='coerce').astype('uint8')
test_set.dropna(subset=['tax_class_at_time_of_sale'], inplace=True)
test_set = pd.get_dummies(test_set, columns=['tax_class_at_time_of_sale'], prefix='tax_class', dtype='uint8')
# BUILDING CLASS AT TIME OF SALE
test_set = pd.get_dummies(test_set, columns=['building_class_at_time_of_sale'], prefix='building_class', drop_first=True, dtype='uint8')
# LAT and LONG
test_set = test_set.dropna(subset=['latitude', 'longitude'])#2384

# SALE DATE
# Load the quarterly house price index data
hpi_data = pd.read_csv("NYSTHPI.csv")
hpi_data['observation_date'] = pd.to_datetime(hpi_data['observation_date'])  # Ensure correct datetime format
hpi_data['year'] = hpi_data['observation_date'].dt.year
# Infer the quarter from the observation_date
hpi_data['quarter'] = (hpi_data['observation_date'].dt.month - 1) // 3 + 1  # Map months to quarters
# Ensure sale_date is in the correct datetime format and drop nulls
test_set['sale_date'] = pd.to_datetime(test_set['sale_date'], errors='coerce')  # Convert to datetime, coerce invalid entries to NaT
test_set.dropna(subset=['sale_date'], inplace=True)  # Drop rows where sale_date is NaT
# Extract year and quarter from sale_date
test_set['sale_year'] = test_set['sale_date'].dt.year
test_set['sale_quarter'] = test_set['sale_date'].dt.quarter
# Merge test_set with HPI data based on year and quarter
test_set = pd.merge(
    test_set,
    hpi_data[['year', 'quarter', 'NYSTHPI']],
    left_on=['sale_year', 'sale_quarter'],
    right_on=['year', 'quarter'],
    how='left'
)
# Rename the merged HPI column
test_set.rename(columns={'NYSTHPI': 'quarterly_price_index'}, inplace=True)
# Drop unnecessary columns
test_set.drop(columns=['sale_quarter', 'year', 'quarter'], inplace=True)
# YEAR BUILT
test_set['year_built'] = pd.to_numeric(test_set['year_built'], errors='coerce')
test_set.dropna(subset=['year_built'], inplace=True)
test_set['year_built'] = test_set['year_built'].astype(int)
test_set = test_set[(test_set['year_built'] >= 1800) & (test_set['year_built'] <= 2024)]
# Calculate house_age at sale
test_set['house_age'] = test_set['sale_year'] - test_set['year_built']
test_set = test_set[test_set['house_age'] >= 0]
test_set.drop(columns=['sale_date', 'sale_year'], inplace=True)
test_set.drop(columns=['year_built'], inplace=True)
# SALE PRICE
# Remove any values with Sale Price that fall in Gift Range or invalid values
test_set["sale_price"] = pd.to_numeric(test_set["sale_price"], errors="coerce")
test_set.dropna(subset=["sale_price"], inplace=True)
# Define borough-specific lower bound rates
borough_lower_bounds = {
    "StatenIsland": 82,
    "Brooklyn": 137,
    "Bronx": 229,
    "Queens": 229,
    "Manhattan": 1740.3
}
# Apply borough-specific lower bound filtering
test_set = test_set[
    test_set.apply(lambda row: row['sale_price'] >= row['quarterly_price_index'] * borough_lower_bounds.get(row['borough'], 82), axis=1)
]

# Apply upper bound filtering (remains the same for all boroughs)
upper_bound = test_set['quarterly_price_index'] * 138000
test_set = test_set[test_set['sale_price'] <= upper_bound]
# DROP LOCATIONS because latitude and longitude was added
test_set = test_set.drop(columns=['borough', 'neighborhood'])

# REMOVING OUTLIERS (Not needed for residential single family homes)
# Residential Units
# Calculate the 99th percentile
# upper_limit = test_set['residential_units'].quantile(0.99)
# test_set['residential_units'] = test_set['residential_units'].clip(upper=upper_limit)
# Commercial Units (No effect on residential housing - keep commented)
# # Calculate the 99th percentile
# upper_limit = test_set['commercial_units'].quantile(0.99)
# test_set['commercial_units'] = test_set['commercial_units'].clip(upper=upper_limit)
# Total Units
# # Calculate the 99th percentile
# upper_limit = test_set['total_units'].quantile(0.99)
# test_set['total_units'] = test_set['total_units'].clip(upper=upper_limit)
# Land Square Feet
# Cap at the 99th percentile to reduce the effect of outliers
# upper_limit = test_set['land_square_feet'].quantile(0.99)
# test_set['land_square_feet'] = test_set['land_square_feet'].clip(upper=upper_limit)
# Gross Square Feet
# Cap at the 99th percentile to reduce the effect of outliers
# upper_limit = test_set['gross_square_feet'].quantile(0.99)
# test_set['gross_square_feet'] = test_set['gross_square_feet'].clip(upper=upper_limit)
# RESIDENTIAL SINGLE FAMILY ONLY SPECIFIC CLEANING
# Apply filters to keep only single-family residential homes with no commercial units
test_set = test_set[
    (test_set['residential_units'] == 1) &
    (test_set['total_units'] == 1) &
    (test_set['commercial_units'] == 0)
]
test_set = test_set[test_set['land_square_feet'] >= test_set['gross_square_feet']] 

# CALCULATE CORRELATION OF FEATURES
# correlation_matrix = test_set.corr()
# target_correlation = correlation_matrix["sale_price"].sort_values(ascending=False)
# target_correlation.to_csv('target_correlation.csv')

# target_correlation = pd.read_csv('target_correlation.csv', index_col=0)
 # Convert to a Series
# if isinstance(target_correlation, pd.DataFrame):
#     target_correlation = target_correlation.iloc[:, 0]
# # Replace NaN or missing values with 0 
# target_correlation = target_correlation.fillna(0)
# # Define a correlation threshold
# correlation_threshold = 0 # 0.01
# # Filter out columns based on the threshold
# columns_to_keep = target_correlation[target_correlation.abs() >= correlation_threshold].index
# columns_to_drop = target_correlation[target_correlation.abs() < correlation_threshold].index

# # Drop columns below the threshold from dataset
# test_set = test_set.drop(columns=columns_to_drop)
# print(f"Dropped columns: {columns_to_drop}")
# print(f"Remaining columns: {columns_to_keep}")
# APPLY LOG TRANSFORMATIONS
test_set['log_sale_price'] = np.log1p(test_set['sale_price'])
test_set['residential_units'] = np.log1p(test_set['residential_units'])
test_set['commercial_units'] = np.log1p(test_set['commercial_units'])
test_set['total_units'] = np.log1p(test_set['total_units'])
test_set['land_square_feet'] = np.log1p(test_set['land_square_feet'])
test_set['gross_square_feet'] = np.log1p(test_set['gross_square_feet'])
# Years Since Sale (Log Transformation Defintely Not Needed)
# House Age (Log Transformation Likely Not Needed)
print(test_set.shape)

# Prepare features and labels
test_features = test_set.drop(columns=['sale_price', 'log_sale_price'])  # Drop original and log-transformed sale prices
test_labels = test_set['log_sale_price']  # Use the log-transformed target
test_labels_original = np.expm1(test_labels)  # Convert log-transformed labels back to original sale prices

model.fit(train_features, train_labels)
test_predictions_log = model.predict(test_features)
test_predictions_original = np.expm1(test_predictions_log)  # Convert predictions back to the original scale
# Calculate metrics for the model
test_mape_original = mean_absolute_percentage_error(test_labels_original, test_predictions_original) * 100
test_accuracy_original = 100 - test_mape_original
test_medae = median_absolute_error(test_labels_original, test_predictions_original)
test_mae = mean_absolute_error(test_labels_original, test_predictions_original)
print(f"Test Set Mean Absolute Percentage Error (MAPE): {test_mape_original:.2f}%")
print(f"Test Set Model Accuracy (MAPE): {test_accuracy_original:.2f}%")
print(f"Test Set Median Absolute Error (MedAE): ${test_medae:,.2f}")
print(f"Test Set Mean Absolute Error (MAE): ${test_mae:,.2f}")
