import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import joblib

print("Loading data...")
df = pd.read_csv('employee_data.csv')

# Drop trailing empty column created by trailing commas
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['Emp_ID'] = ['EMP' + str(i).zfill(3) for i in range(1, len(df) + 1)]

# Drop target and leaky columns
X_raw = df.drop(columns=['Attrition', 'Date_of_Hire', 'Date_of_termination', 'Status_of_leaving', 'Emp_ID'], errors='ignore')

# Convert target
y = df['Attrition'].map({'Yes': 1, 'No': 0})

# One-hot encoding
print("Encoding data...")
X = pd.get_dummies(X_raw, drop_first=True)

# Save processed dataframe to avoid re-encoding in Streamlit
df_processed = pd.concat([df[['Emp_ID', 'Attrition']], X], axis=1)
df_processed.to_csv('employee_data_processed.csv', index=False)

print("Training model...")
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

# Train a million times (simulated with randomized search for robustness)
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, cv=3, scoring='accuracy', random_state=42)
search.fit(X, y)

print("Best parameters found:", search.best_params_)

print("Saving model...")
joblib.dump(search.best_estimator_, 'attrition_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')
print("Model training complete!")
