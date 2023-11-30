import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Load the data
df = pd.read_csv('Car_features.csv')

# Ensure consistency in column naming and preprocessing
df.columns = df.columns.str.lower().str.replace(' ', '_')

# If 'msrp' column exists, rename it to 'price' and create 'log_price'
if 'msrp' in df.columns:
    df = df.rename(columns={'msrp': 'price'})
    df['log_price'] = np.log1p(df['price'])
elif 'price' in df.columns:  # If 'price' column exists, create 'log_price'
    df['log_price'] = np.log1p(df['price'])
else:
    raise ValueError("Price column not found in dataset")

# Identify string columns and preprocess them
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Preprocessing
df = pd.get_dummies(df, columns=string_columns)
df.fillna(df.mean(), inplace=True)

X = df.drop('log_price', axis=1)
y = df['log_price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# After preprocessing
expected_columns = df.columns.tolist()
print(expected_columns)  # Use this output in your Streamlit app


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.3, random_state=42)

# Train the model
model = Ridge(alpha=.1)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'ridge_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
