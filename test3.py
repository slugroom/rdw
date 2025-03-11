import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
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
# 5. Create a Pipeline with StandardScaler and XGBoost
# ---------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42))
])

# ---------------------------
# 6. Hyperparameter Tuning with Optuna
# ---------------------------
def objective(trial):
    # Define hyperparameters to search
    params = {
        'clf__n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
        'clf__max_depth': trial.suggest_categorical('max_depth', [3, 5, 7]),
        'clf__learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.1, 0.2]),
        'clf__subsample': trial.suggest_categorical('subsample', [0.8, 1.0]),
        'clf__colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.8, 1.0])
    }
    # Set the parameters in the pipeline
    pipeline.set_params(**params)
    
    # Use cross-validation to evaluate the current hyperparameter set
    score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1).mean()
    return score

# Create an Optuna study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print best hyperparameters found
print("Best trial:")
print(study.best_trial.params)

# ---------------------------
# 7. Train the Final Model using Best Parameters
# ---------------------------
# Set the best parameters into the pipeline and train on the full training set
pipeline.set_params(
    clf__n_estimators=study.best_trial.params['n_estimators'],
    clf__max_depth=study.best_trial.params['max_depth'],
    clf__learning_rate=study.best_trial.params['learning_rate'],
    clf__subsample=study.best_trial.params['subsample'],
    clf__colsample_bytree=study.best_trial.params['colsample_bytree']
)
pipeline.fit(X_train, y_train)

# ---------------------------
# 8. Evaluate the Model
# ---------------------------
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test ROC AUC:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))

# ---------------------------
# 9. Predicting Win Probabilities for a New Race
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
win_probs = pipeline.predict_proba(X_new)[:, 1]
new_drivers['win_probability'] = win_probs

# Sort by win probability descending and print the results
new_drivers = new_drivers.sort_values(by='win_probability', ascending=False)
print(new_drivers[['driver_id', 'driver_name', 'win_probability']])

# ---------------------------
# 10. Save Predictions to a CSV File
# ---------------------------
new_drivers[['driver_id', 'driver_name', 'win_probability']].to_csv("predictions.csv", index=False)
