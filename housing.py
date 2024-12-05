import pandas as pd
import numpy as np
import janitor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_name = "nyc-property-sales.csv"
data = pd.read_csv(file_name)


data = data.clean_names()

# Split data into test and train set
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# DROP COLUMNS
train_set = train_set.drop(columns=['tax_class_at_present']) # do not care about present tax class
train_set = train_set.drop(columns=['block']) # too many values (19,456) to use in regression model
train_set = train_set.drop(columns=['lot']) # too many values (7,767) inconsequential
train_set = train_set.drop(columns=['building_class_at_present']) # do not care about present building class
train_set = train_set.drop(columns=['address']) # don't care about street name or number
train_set = train_set.drop(columns=['apartment_number']) # apartment number has very little value except for floor
train_set = train_set.drop(columns=['_zip_code']) # borough covers the larger areas and neighborhood already covers smaller areas + too many values

# BOROUGH COLUMN
# Convert to numbers and drop rows with null column values
print(train_set.shape)
train_set["borough"] = pd.to_numeric(train_set["borough"], errors="coerce")
train_set.dropna(subset=["borough"], inplace=True) # dropped one row

train_set = pd.get_dummies(train_set, columns=["borough"], prefix="borough", dtype='uint8') # one hot encoding



# NEIGHBORHOOD COLUMN
# Trim white space and standardize the text to uppercase
train_set['neighborhood'] = train_set['neighborhood'].str.strip().str.upper()
train_set['neighborhood'] = train_set['neighborhood'].str.replace(r'\s+', ' ', regex=True) # added, recheck later

# Remove rows where 'neighborhood' contains numeric values or invalid patterns
train_set = train_set[~train_set['neighborhood'].str.isnumeric()]  # Remove purely numeric neighborhoods
# Define specific invalid patterns to remove
invalid_patterns = ['-UNKNOWN', 'NEIGHBORHOOD']

# # Remove rows where 'neighborhood' contains the invalid patterns (dropped 200 rows)
train_set = train_set[~train_set['neighborhood'].str.contains('|'.join(invalid_patterns), case=False, na=False)]

train_set = pd.get_dummies(train_set, columns=['neighborhood'], prefix='neighborhood', dtype='uint8')




# BUILDING CLASS CATEGORY: 
train_set['building_class_category'] = train_set['building_class_category'].str.strip().str.upper()
# Replace multiple spaces with a single space
train_set['building_class_category'] = train_set['building_class_category'].str.replace(r'\s+', ' ', regex=True)

category_mapping = {
    "01 ONE FAMILY DWELLINGS": "ONE-FAMILY RESIDENTIAL",
    "01 ONE FAMILY HOMES": "ONE-FAMILY RESIDENTIAL",
    "02 TWO FAMILY DWELLINGS": "TWO-FAMILY RESIDENTIAL",
    "02 TWO FAMILY HOMES": "TWO-FAMILY RESIDENTIAL",
    "03 THREE FAMILY DWELLINGS": "THREE-FAMILY RESIDENTIAL",
    "03 THREE FAMILY HOMES": "THREE-FAMILY RESIDENTIAL",
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

# Perform one-hot encoding for 'building_class_category'
# One hot encoding for building class category is done in the land square feet section


# EASEMENT:
# Replace 'E' and 'EASEMENT' with 1, and everything else with 0
train_set['ease_ment'] = train_set['ease_ment'].replace({'E': 1, 'EASEMENT': 1})
# Convert empty strings or any non-numeric value to 0
train_set['ease_ment'] = train_set['ease_ment'].replace(' ', 0)
# Fill NaN with 0 and convert to integers
train_set['ease_ment'] = train_set['ease_ment'].fillna(0).astype(int)


# RESIDENTIAL UNITS
train_set = train_set[~train_set['residential_units'].isin(['RESIDENTIAL UNITS'])]
train_set['residential_units'] = pd.to_numeric(train_set['residential_units'], errors='coerce')
train_set.dropna(subset=['residential_units'], inplace=True) # drops 100000 rows
train_set['residential_units'] = train_set['residential_units'].astype(int)


# COMMERCIAL UNITS
train_set = train_set[~train_set['commercial_units'].isin(['COMMERCIAL UNITS'])]  # Remove header-like rows
train_set['commercial_units'] = pd.to_numeric(train_set['commercial_units'], errors='coerce')  # Convert to numeric
train_set['commercial_units'].fillna(0, inplace=True)  # Fill missing values with 0
train_set['commercial_units'] = train_set['commercial_units'].astype(int)  # Convert to integers


# TOTAL UNITS:
# Replace 'TOTAL UNITS' with numeric values and handle invalid entries
train_set = train_set[~train_set['total_units'].isin(['TOTAL UNITS'])]  # Remove header-like rows
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


# GROSS SQUARE FEET:
train_set['gross_square_feet'] = pd.to_numeric(train_set['gross_square_feet'], errors='coerce')

train_set.dropna(subset=['gross_square_feet'], inplace=True) # drops around 50000 rows



# YEAR BUILT
train_set['year_built'] = pd.to_numeric(train_set['year_built'], errors='coerce')
train_set.dropna(subset=['year_built'], inplace=True)
train_set['year_built'] = train_set['year_built'].astype(int)
train_set = train_set[(train_set['year_built'] >= 1800) & (train_set['year_built'] <= 2024)]
# drops around 140,000 total rows


# TAX CLASS AT TIME OF SALE
train_set['tax_class_at_time_of_sale'] = pd.to_numeric(train_set['tax_class_at_time_of_sale'], errors='coerce').astype('uint8')
train_set.dropna(subset=['tax_class_at_time_of_sale'], inplace=True) # nothing dropped
train_set = pd.get_dummies(train_set, columns=['tax_class_at_time_of_sale'], prefix='tax_class', dtype='uint8')




# BUILDING CLASS AT TIME OF SALE
train_set = pd.get_dummies(train_set, columns=['building_class_at_time_of_sale'], prefix='building_class', drop_first=True, dtype='uint8')





# SALE PRICE
# Remove any values with Sale Price of 0 or invalid values
train_set["sale_price"] = pd.to_numeric(train_set["sale_price"], errors="coerce")
# print(train_set["sale_price"].describe())
train_set.dropna(subset=["sale_price"], inplace=True)
train_set = train_set[train_set["sale_price"] > 10000] # drops 500,000 rows - 400,000 after previous drops

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = train_set['sale_price'].quantile(0.25)
Q3 = train_set['sale_price'].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset to remove outliers
train_set = train_set[(train_set['sale_price'] >= lower_bound) & (train_set['sale_price'] <= upper_bound)]

# Apply log transformation to sale_price
train_set['log_sale_price'] = np.log1p(train_set['sale_price'])

 # Plot the distribution of sale_price
# plt.figure(figsize=(10, 6))
# sns.histplot(train_set['sale_price'], bins=50, kde=True, color='blue')
# plt.title('Distribution of Sale Price', fontsize=16)
# plt.xlabel('Sale Price', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# Print basic statistics for sale_price
# print("Basic Statistics for Sale Price:")
# print(train_set['sale_price'].describe())


# SALE DATE
# Ensure sale_date is in the correct datetime format and drop nulls
train_set['sale_date'] = pd.to_datetime(train_set['sale_date'], errors='coerce')  # Convert to datetime, coerce invalid entries to NaT
train_set.dropna(subset=['sale_date'], inplace=True)  # Drop rows where sale_date is NaT

# Extract sale_year from sale_date
train_set['sale_year'] = train_set['sale_date'].dt.year


# Create a new column 'years_since_sale'
current_year = 2024  # Replace with the desired current year
train_set['years_since_sale'] = current_year - train_set['sale_year']

# Drop sale_year column as it's no longer needed
# Drop columns no longer needed
train_set.drop(columns=['sale_date', 'sale_year'], inplace=True)


# correlation_matrix = train_set.corr()
# target_correlation = correlation_matrix["sale_price"].sort_values(ascending=False)
# target_correlation.to_csv('target_correlation.csv')


"""" NORMALIZATION """
# Residential Units
# Calculate the 99th percentile
upper_limit = train_set['residential_units'].quantile(0.99)
# Cap values
train_set['residential_units'] = train_set['residential_units'].clip(upper=upper_limit)
# Apply log transformation
train_set['residential_units'] = np.log1p(train_set['residential_units'])
# Scale the feature
scaler = MinMaxScaler()
train_set['residential_units'] = scaler.fit_transform(train_set[['residential_units']])

# Commercial Units
# Calculate the 99th percentile
upper_limit = train_set['commercial_units'].quantile(0.99)
# Cap values
train_set['commercial_units'] = train_set['commercial_units'].clip(upper=upper_limit)
# Apply log transformation
train_set['commercial_units'] = np.log1p(train_set['commercial_units'])
# Scale the feature
train_set['commercial_units'] = scaler.fit_transform(train_set[['commercial_units']])

# Total Units
# Calculate the 99th percentile
upper_limit = train_set['total_units'].quantile(0.99)
# Cap values
train_set['total_units'] = train_set['total_units'].clip(upper=upper_limit)
# Apply log transformation
train_set['total_units'] = np.log1p(train_set['total_units'])
# Scale the feature
train_set['total_units'] = scaler.fit_transform(train_set[['total_units']])

# Land Square Feet
# Cap at the 99th percentile to reduce the effect of outliers
upper_limit = train_set['land_square_feet'].quantile(0.99)
train_set['land_square_feet'] = train_set['land_square_feet'].clip(upper=upper_limit)

# Apply log transformation to reduce skewness
train_set['land_square_feet'] = np.log1p(train_set['land_square_feet'])

# Scale the data to normalize values between 0 and 1
train_set['land_square_feet'] = scaler.fit_transform(train_set[['land_square_feet']])

# Gross Square Feet (50 % of data is 0? Should drop)
# Cap at the 99th percentile to reduce the effect of outliers
upper_limit = train_set['gross_square_feet'].quantile(0.99)
train_set['gross_square_feet'] = train_set['gross_square_feet'].clip(upper=upper_limit)

# Apply log transformation to reduce skewness
train_set['gross_square_feet'] = np.log1p(train_set['gross_square_feet'])

# Scale the data to normalize values between 0 and 1
train_set['gross_square_feet'] = scaler.fit_transform(train_set[['gross_square_feet']])


# SALE YEAR
train_set['years_since_sale'] = np.log1p(train_set['years_since_sale'])
train_set['years_since_sale'] = scaler.fit_transform(train_set[['years_since_sale']])

# Display basic statistics for years_since_sale
print("Basic Statistics for years_since_sale:")
print(train_set['years_since_sale'].describe())

# Check for NaN or infinite values
print("\nNaN values in years_since_sale:", train_set['years_since_sale'].isna().sum())
print("Infinite values in years_since_sale:", np.isinf(train_set['years_since_sale']).sum())

# Display unique values and their counts (if necessary)
print("\nUnique values in years_since_sale:")
print(train_set['years_since_sale'].value_counts())

# Display a sample of the column to inspect the data
print("\nSample of years_since_sale values:")
print(train_set['years_since_sale'].head(10))

target_correlation = pd.read_csv('target_correlation.csv', index_col=0)



# Convert to a Series
if isinstance(target_correlation, pd.DataFrame):
    target_correlation = target_correlation.iloc[:, 0]

# Replace NaN or missing values with 0 
target_correlation = target_correlation.fillna(0)

# Define a correlation threshold
correlation_threshold = 0.008

# Filter out columns based on the threshold
columns_to_keep = target_correlation[target_correlation.abs() >= correlation_threshold].index
columns_to_drop = target_correlation[target_correlation.abs() < correlation_threshold].index



# Drop columns below the threshold from your dataset
train_set = train_set.drop(columns=columns_to_drop)



print(f"Dropped columns: {columns_to_drop}")
print(f"Remaining columns: {columns_to_keep}")
print(train_set.shape)


# train_set = train_set[train_set['gross_square_feet'] != 0]
# Prepare features and labels
features = train_set.drop(columns=['sale_price', 'log_sale_price'])  # Drop both original and log-transformed
# Convert scaled features back to a DataFrame for inspection
# features_df = pd.DataFrame(train_set, columns=train_set.drop(columns=['sale_price', 'log_sale_price']).columns)

# # Convert scaled features back to a DataFrame for inspection
scaler = StandardScaler()
features = scaler.fit_transform(features)
# # Display the first few rows of the features
# print("Sample of Features:")
# print(features_df.head())

# # Display descriptive statistics of the features
# print("\nDescriptive Statistics of Features:")
# print(features_df.describe())


labels = train_set['log_sale_price']  # Use log-transformed sale_price

# TRAINING
# Initialize the model
model = LinearRegression()



# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get predictions
predictions_log = cross_val_predict(model, features, labels, cv=kf)

# Inverse the log transformation to interpret results on original scale
print("Range of predictions_log:")
print(f"Min: {predictions_log.min()}, Max: {predictions_log.max()}")

print("Features statistics:")
print(pd.DataFrame(features).describe())
print("\nLabels statistics:")
print(labels.describe())

predictions = np.expm1(predictions_log)  # Inverse transformation
actual_values = np.expm1(labels)  # Convert log labels back to original scale

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(actual_values, predictions) * 100

# Calculate Accuracy
accuracy = 100 - mape

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Model Accuracy: {accuracy:.2f}%")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(actual_values, predictions)

print(f"Mean Absolute Error (MAE): ${mae:,.2f}")



# TEST
