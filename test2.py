import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------
# 1. Load the CSV Data
# ---------------------------
drivers_df = pd.read_csv('vehicles.csv')
track_df = pd.read_csv('track.csv')
race_results_df = pd.read_csv('results.csv')

# ---------------------------
# 2. Merge Race Results with Driver/Vehicle Data
# ---------------------------
merged_df = race_results_df.merge(drivers_df, left_on='driverId', right_on='driver_id')
merged_df['win'] = (merged_df['position'] == 1).astype(int)

# ---------------------------
# 3. Feature Engineering & Preprocessing
# ---------------------------
numeric_features = ['massa_rijklaar', 'cilinderinhoud', 'vermogen_massarijklaar', 'years_of_experience']
merged_df[numeric_features] = merged_df[numeric_features].apply(pd.to_numeric, errors='coerce')
merged_df[numeric_features] = merged_df[numeric_features].fillna(merged_df[numeric_features].median())

# One-hot encode the driver's driving style
driving_style_encoded = pd.get_dummies(merged_df['driving_style'], prefix='style')
X_driver = pd.concat([merged_df[numeric_features], driving_style_encoded], axis=1)

# --------------------------------------------------
# Incorporate Track Features into the Model
# --------------------------------------------------
# Drop non-feature columns from track data
track_features = track_df.drop(['id', 'name'], axis=1)

# One-hot encode categorical track features (weather and time_of_day)
track_cat = track_features[['weather', 'time_of_day']]
track_num = track_features.drop(['weather', 'time_of_day'], axis=1)
track_cat_encoded = pd.get_dummies(track_cat, prefix=['weather', 'time'])
track_feat_single = pd.concat([track_num, track_cat_encoded], axis=1)

# Since the track is the same for all drivers in a race,
# replicate the track feature row to match the number of rows in our training data.
track_feat = pd.concat([track_feat_single] * len(merged_df), ignore_index=True)

# Combine driver/vehicle features with track features
X = pd.concat([X_driver.reset_index(drop=True), track_feat], axis=1)
y = merged_df['win']

# ---------------------------
# 4. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 5. Create a Pipeline with StandardScaler and CatBoost
# ---------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', CatBoostClassifier(
        iterations=100,       # default value; will be tuned by GridSearchCV
        depth=6,              # default value; will be tuned
        learning_rate=0.1,      # default value; will be tuned
        verbose=0,
        random_seed=42))
])

# ---------------------------
# 6. Hyperparameter Tuning via GridSearchCV
# ---------------------------
param_grid = {
    'clf__iterations': [50, 100, 200],
    'clf__depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__bagging_temperature': [0.8, 1]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

# ---------------------------
# 7. Evaluate the Model
# ---------------------------
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test ROC AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))

# ---------------------------
# 8. Predicting Win Probabilities for a New Race
# ---------------------------
new_drivers = drivers_df.copy()

# Preprocess new driver data
new_drivers[numeric_features] = new_drivers[numeric_features].apply(pd.to_numeric, errors='coerce')
new_drivers[numeric_features] = new_drivers[numeric_features].fillna(new_drivers[numeric_features].median())
new_style = pd.get_dummies(new_drivers['driving_style'], prefix='style')
X_new_driver = pd.concat([new_drivers[numeric_features], new_style], axis=1)

# Ensure the new data has the same columns as the training driver features.
for col in X_driver.columns:
    if col not in X_new_driver.columns:
        X_new_driver[col] = 0
X_new_driver = X_new_driver[X_driver.columns]

# Add track features for the new race
track_feat_new = pd.concat([track_feat_single] * len(new_drivers), ignore_index=True)
X_new = pd.concat([X_new_driver.reset_index(drop=True), track_feat_new.reset_index(drop=True)], axis=1)
X_new = X_new[X.columns]

# Predict win probabilities (probability that win == 1)
win_probs = best_model.predict_proba(X_new)[:, 1]
new_drivers['win_probability'] = win_probs

# Sort by win probability descending
new_drivers = new_drivers.sort_values(by='win_probability', ascending=False)
print(new_drivers[['driver_id', 'driver_name', 'win_probability']])

# ---------------------------
# 9. Save Predictions to a CSV File
# ---------------------------
new_drivers[['driver_id', 'driver_name', 'win_probability']].to_csv("predictions2.csv", index=False)
