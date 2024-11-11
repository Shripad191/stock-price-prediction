import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle

# Load the data
data = pd.read_csv('./data.csv')

# Display the first few rows to understand the data
print(data.head())

# Drop rows with missing values (if any)
data.dropna(inplace=True)

# Feature engineering
# Drop 'Volume' and 'Turnover' columns, as per your requirement
X = data[['Prev Close', 'Open', 'High', 'Low']]  # Removed 'Volume' and 'Turnover'
y = data['Close']  # The stock price we want to predict

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
# For now, we will standardize numerical features using StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns)  # Apply scaling to all features
    ])

# Create a pipeline for each model
models = {
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    'Linear Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ]),
    'Decision Tree': Pipeline([
        ('preprocessor', preprocessor),
        ('model', DecisionTreeRegressor(random_state=42))
    ])
}

# Train all models and pick the best one based on performance
best_model = None
best_score = -np.inf
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # R-squared score for regression
    print(f'{name} Model Score: {score}')
    
    if score > best_score:
        best_score = score
        best_model = model

# Save the best model using pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"Best model is: {best_model.named_steps['model'].__class__.__name__}")
