import pandas as pd
import numpy as np
import janitor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# Load the dataset
file_name = "nyc-property-sales.csv"
data = pd.read_csv(file_name)


data = data.clean_names()

# print(data.info())
# print(data.describe())


# print(corr_matrix["SALE PRICE"].sort_values(ascending=False))


train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# BOROUGH COLUMN
# Convert to numbers and drop rows with null column values
# train_set["borough"] = pd.to_numeric(train_set["borough"], errors="coerce")
# train_set.dropna(subset=["borough"], inplace=True)

# train_set = pd.get_dummies(train_set, columns=["borough"], prefix="borough") # one hot encoding


# NEIGHBORHOOD COLUMN
# Trim white space and standardize the text to uppercase
# train_set['neighborhood'] = train_set['neighborhood'].str.strip().str.upper()
# train_set['neighborhood'] = train_set['neighborhood'].str.replace(r'\s+', ' ', regex=True) # added, recheck later

# # Remove rows where 'neighborhood' contains numeric values or invalid patterns
# train_set = train_set[~train_set['neighborhood'].str.isnumeric()]  # Remove purely numeric neighborhoods
# # Define specific invalid patterns to remove
# invalid_patterns = ['-UNKNOWN', 'NEIGHBORHOOD']

# # Remove rows where 'neighborhood' contains the invalid patterns
# train_set = train_set[~train_set['neighborhood'].str.contains('|'.join(invalid_patterns), case=False, na=False)]

# # Display unique neighborhoods to confirm the cleaning process
# print(train_set['neighborhood'].unique())
# print(f"Number of unique neighborhoods: {len(train_set['neighborhood'].unique())}")

# train_set = pd.get_dummies(train_set, columns=['neighborhood'], prefix='neighborhood')
# print(f"Number of columns after encoding: {train_set.shape[1]}")



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
train_set = train_set[train_set['building_class_category'].isin(valid_categories)]
print(train_set["building_class_category"].unique())
print(len(train_set['building_class_category'].unique()))
# Perform one-hot encoding for 'building_class_category'
# One hot encoding for building class category is done in the land square feet section




# print(f"Number of rows: {len(train_set)}")

# DROP COLUMNS
# train_set = train_set.drop(columns=['tax_class_at_present']) # do not care about present tax class
# train_set = train_set.drop(columns=['block']) # too many values (19,456) to use in regression model
# train_set = train_set.drop(columns=['lot']) # too many values (7,767) inconsequential
# train_set = train_set.drop(columns=['building_class_at_present']) # do not care about present building class
# train_set = train_set.drop(columns=['address']) # don't care about street name or number
# train_set = train_set.drop(columns=['apartment_number']) # apartment number has very little value except for floor
# train_set = train_set.drop(columns=['_zip_code']) # borough covers the larger areas and neighborhood already covers smaller areas + too many values


# EASEMENT:
# Replace 'E' and 'EASEMENT' with 1, and everything else with 0
# train_set['ease_ment'] = train_set['ease_ment'].replace({'E': 1, 'EASEMENT': 1})
# # Convert empty strings or any non-numeric value to 0
# train_set['ease_ment'] = train_set['ease_ment'].replace(' ', 0)
# # Fill NaN with 0 and convert to integers
# train_set['ease_ment'] = train_set['ease_ment'].fillna(0).astype(int)

# RESIDENTIAL UNITS
# train_set = train_set[~train_set['residential_units'].isin(['RESIDENTIAL UNITS'])]
# train_set['residential_units'] = pd.to_numeric(train_set['residential_units'], errors='coerce')
# train_set.dropna(subset=['residential_units'], inplace=True) # drops 10000 rows
# train_set['residential_units'] = train_set['residential_units'].astype(int)

# COMMERICIAL UNITS

# train_set = train_set[~train_set['commercial_units'].isin(['COMMERCIAL UNITS'])]  # Remove header-like rows
# train_set['commercial_units'] = pd.to_numeric(train_set['commercial_units'], errors='coerce')  # Convert to numeric
# train_set['commercial_units'].fillna(0, inplace=True)  # Fill missing values with 0
# train_set['commercial_units'] = train_set['commercial_units'].astype(int)  # Convert to integers

# print(train_set['commercial_units'].unique())
# print(len(train_set['commercial_units'].unique()))


# TOTAL UNITS:

# Replace 'TOTAL UNITS' with numeric values and handle invalid entries
# train_set = train_set[~train_set['total_units'].isin(['TOTAL UNITS'])]  # Remove header-like rows
# train_set['total_units'] = pd.to_numeric(train_set['total_units'], errors='coerce')  # Convert to numeric
# # print(train_set.shape[0])
# train_set.dropna(subset=['total_units'], inplace=True)  # Drop rows with NaN values (9000 affected)
# # print(train_set.shape[0])
# train_set['total_units'] = train_set['total_units'].astype(int)  # Convert to integers

# LAND SQUARE FEET:
# condo_types = ['ELEVATOR APARTMENTS', 'WALKUP APARTMENTS', 'SMALL CONDOS', 'CONDO COOPS AND CONDOPS']
# train_set.loc[train_set['building_class_category'].isin(condo_types), 'land_square_feet'] = train_set.loc[
#     train_set['building_class_category'].isin(condo_types), 'land_square_feet'].fillna(0)
# train_set['land_square_feet'] = pd.to_numeric(train_set['land_square_feet'], errors='coerce')
# train_set.dropna(subset=['land_square_feet'], inplace=True) # drops 1,000 rows if condos are assigned 0 
# train_set = pd.get_dummies(train_set, columns=['building_class_category'], prefix='class') # does one hot encoding for building class category

# GROSS SQUARE FEET:

print(train_set['land_square_feet'].unique())
print(len(train_set['land_square_feet'].unique()))

# SALE PRICE
# Remove any values with Sale Price of 0 or invalid values
# train_set["SALE PRICE"] = pd.to_numeric(train_set["SALE PRICE"], errors="coerce")
# train_set.dropna(subset=["SALE PRICE"], inplace=True)
# train_set = train_set[train_set["SALE PRICE"] > 0]






# numerical_data = train_set.select_dtypes(include=['number'])

# corr_matrix = numerical_data.corr()


# print(corr_matrix["SALE PRICE"].sort_values(ascending=False))

# TRAINING 

# TEST
