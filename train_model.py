import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('deposit term dirty null.csv', low_memory=False)

df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
df = df.drop(['Id', 'BankId', 'Year', 'first_name', 'last_name', 'email'], axis=1, errors='ignore')
df.dropna(subset=['housing', 'default',  'month', 'contact', 'job'], inplace=True)
df['age'] = df['age'].fillna(df['age'].median())    # Filing the null values with median
df['balance'] = df['balance'].fillna(df['balance'].median()) # Filing the null values with median
df['duration'] = df['duration'].fillna(df['duration'].median()) # Filing the null values with median
df['pdays'] = df['pdays'].fillna(df['pdays'].median()) # Filing the null values with median

mean_balance = df[df['balance'] >= 0]['balance'].mean()
df['balance'] = df['balance'].apply(lambda x: mean_balance if x < 0 else x)

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    return df

# Example usage
df = cap_outliers(df, 'age')
df = cap_outliers(df, 'balance')
df = cap_outliers(df, 'day')
df = cap_outliers(df, 'campaign')
# Repeat for other columns as needed

df = df.drop(['pdays','previous'], axis=1)


# Check for and remove non-numeric entries in numeric columns
numeric_columns = ['balance', 'day', 'campaign', 'duration']  # Update if needed

# Function to convert columns to numeric, forcing errors to NaN
def to_numeric(column):
    return pd.to_numeric(column, errors='coerce')

for col in numeric_columns:
    df[col] = to_numeric(df[col])

# Fill missing values for numeric columns
for col in numeric_columns:
    df[col].fillna(df[col].median())

# Fill missing values for categorical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0])

# Convert the target column 'y' to binary values (0 = no, 1 = yes)
df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Drop any remaining rows with NaN values
df.dropna(inplace=True)

# Split the data into features (X) and target (y)
X = df.drop('y', axis=1)
y = df['y']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define preprocessing pipeline: scale numeric and one-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Define the RandomForest model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

import joblib  # Import joblib for saving models
# other imports...
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming model_pipeline is already created and trained

# Save the model and preprocessor using joblib
joblib.dump(model_pipeline, 'model.joblib') 

print("Model saved!")
