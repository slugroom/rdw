import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. Load the CSV Data
# ---------------------------
# Replace the filenames with your actual file paths
drivers_df = pd.read_csv('vehicles.csv')
track_df = pd.read_csv('track.csv')
race_results_df = pd.read_csv('results.csv')

# ---------------------------
# 2. Merge Race Results with Driver/Vehicle Data
# ---------------------------
# Merge on driver ID (note: adjust column names if needed)
merged_df = race_results_df.merge(drivers_df, left_on='driverId', right_on='driver_id')

# Create a binary target variable: 1 if driver finished first (winner), else 0
merged_df['win'] = (merged_df['position'] == 1).astype(int)

# ---------------------------
# 3. Feature Engineering & Preprocessing
# ---------------------------
# Select some numeric features from vehicle/driver info.
numeric_features = ['massa_rijklaar', 'cilinderinhoud', 'vermogen_massarijklaar', 'years_of_experience']

# Handle missing values (fill with median as an example)
merged_df[numeric_features] = merged_df[numeric_features].apply(pd.to_numeric, errors='coerce')
merged_df[numeric_features] = merged_df[numeric_features].fillna(merged_df[numeric_features].median())

# One-hot encode the driver's driving style
driving_style_encoded = pd.get_dummies(merged_df['driving_style'], prefix='style')

# Combine these features into a feature DataFrame
X_driver = pd.concat([merged_df[numeric_features], driving_style_encoded], axis=1)

# --------------------------------------------------
# Incorporate Track Features into the Model
# --------------------------------------------------
# Drop columns that are not features (like id and name)
track_features = track_df.drop(['id', 'name'], axis=1)

# One-hot encode categorical track features (weather and time_of_day)
track_cat = track_features[['weather', 'time_of_day']]
track_num = track_features.drop(['weather', 'time_of_day'], axis=1)
track_cat_encoded = pd.get_dummies(track_cat, prefix=['weather', 'time'])

# Combine numeric and encoded categorical track features
track_feat_single = pd.concat([track_num, track_cat_encoded], axis=1)

# Since the track is the same for all drivers in a race,
# replicate the track feature row to match the number of rows in our training data.
track_feat = pd.concat([track_feat_single]*len(merged_df), ignore_index=True)

# Combine driver/vehicle features with track features
X = pd.concat([X_driver.reset_index(drop=True), track_feat], axis=1)

# The target variable is:
y = merged_df['win']

# ---------------------------
# 4. Train an XGBoost Classifier
# ---------------------------
# We use a binary:logistic objective since we're predicting win (1) or not (0)
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model.fit(X, y)

# Optionally, check training accuracy (though with very few wins, accuracy may be misleading)
y_pred = model.predict(X)
print("Training Accuracy:", accuracy_score(y, y_pred))

# ---------------------------
# 5. Predicting Win Probabilities for a New Race
# ---------------------------
# For prediction, we use the complete driver/vehicle dataset (all drivers)
new_drivers = drivers_df.copy()

# Ensure numeric columns are numeric and fill missing values
new_drivers[numeric_features] = new_drivers[numeric_features].apply(pd.to_numeric, errors='coerce')
new_drivers[numeric_features] = new_drivers[numeric_features].fillna(new_drivers[numeric_features].median())

# One-hot encode driving style in the new data
new_style = pd.get_dummies(new_drivers['driving_style'], prefix='style')
X_new_driver = pd.concat([new_drivers[numeric_features], new_style], axis=1)

# Ensure the new data has the same columns as the training driver features.
for col in X_driver.columns:
    if col not in X_new_driver.columns:
        X_new_driver[col] = 0
X_new_driver = X_new_driver[X_driver.columns]

# Add the same track features (replicate the single row for every driver)
track_feat_new = pd.concat([track_feat_single]*len(new_drivers), ignore_index=True)
X_new = pd.concat([X_new_driver.reset_index(drop=True), track_feat_new.reset_index(drop=True)], axis=1)

# Make sure the order of columns matches the training data.
X_new = X_new[X.columns]

# Predict win probabilities (probability that win == 1)
win_probs = model.predict_proba(X_new)[:, 1]
new_drivers['win_probability'] = win_probs

# Sort by win probability descending to see the most likely winner at the top
new_drivers = new_drivers.sort_values(by='win_probability', ascending=False)
print(new_drivers[['driver_id', 'driver_name', 'win_probability']])

# ---------------------------
# 6. Save Predictions to a CSV File
# ---------------------------
new_drivers[['driver_id', 'driver_name', 'win_probability']].to_csv("predictions.csv", index=False)
