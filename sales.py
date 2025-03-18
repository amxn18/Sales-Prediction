import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


salesData = pd.read_csv('salesData.csv')
# Handling Missing Values

#  Categorical features 
# 1) Item_Identifier
# 2) Item_Fat_Content
# 3) Item_Type
# 4) Outlet_Identifier
# 5) Outlet_Size
# 6) Outlet_Location_Type
# 7) Outlet_Type


# 1) Fill missing Item_Weight with mean
salesData['Item_Weight'].fillna(salesData['Item_Weight'].mean(), inplace=True)

# 2) Fill missing Outlet_Size based on mode of Outlet_Type
modeOutletSize = salesData.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
missing = salesData['Outlet_Size'].isnull()
salesData.loc[missing, 'Outlet_Size'] = salesData.loc[missing, 'Outlet_Type'].apply(lambda x: modeOutletSize[x][0])


# Data Cleaning & Label Encoding


# Standardize Fat Content labels
salesData.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF':'Low Fat','reg':'Regular'}}, inplace=True)

# Label encode categorical features
encoder = LabelEncoder()
for col in ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']:
    salesData[col] = encoder.fit_transform(salesData[col])


# Feature Selection and Transformation

salesData.drop(['Item_Identifier'], axis=1, inplace=True)

# Log-transform skewed columns
salesData['Item_Visibility'] = np.log1p(salesData['Item_Visibility'])
salesData['Item_Outlet_Sales'] = np.log1p(salesData['Item_Outlet_Sales'])

# Separate features and target
X = salesData.drop('Item_Outlet_Sales', axis=1)
y = salesData['Item_Outlet_Sales']


# Model Training


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = XGBRegressor()
model.fit(x_train, y_train)

# Evaluate Training
train_pred = model.predict(x_train)
train_score = metrics.r2_score(y_train, train_pred)
print("R2 Training Score:", train_score)

# Evaluate Testing
test_pred = model.predict(x_test)
test_score = metrics.r2_score(y_test, test_pred)
print("R2 Test Score:", test_score)


# Predictive System


input_data = (
    9.3,     # Item_Weight
    1217.0,  # Item_Visibility (will be log1p-transformed)
    249.8,   # Item_MRP
    1,       # Item_Fat_Content (Label encoded)
    10,      # Item_Type (Label encoded)
    3,       # Outlet_Identifier (Label encoded)
    1,       # Outlet_Establishment_Year
    1,       # Outlet_Size (Label encoded)
    1,       # Outlet_Location_Type (Label encoded)
    1        # Outlet_Type (Label encoded)
)

input_array = np.asarray(input_data)
input_reshaped = input_array.reshape(1, -1)


input_reshaped[0][1] = np.log1p(input_reshaped[0][1])  


prediction = model.predict(input_reshaped)
output = np.expm1(prediction)  

print("Predicted Sales for given input: ₹", round(output[0], 2))
