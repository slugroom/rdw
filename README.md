# RDW Virtual Racing Betting Optimizer

A data-driven betting strategy system for the RDW Virtual Racing competition that uses machine learning to predict race outcomes and optimize bet placement.

## Overview

This project combines multiple machine learning models to analyze driver, vehicle, and track data from the RDW Virtual Racing API. It generates win probability predictions for each driver in upcoming races and determines an optimal betting strategy to maximize returns.

The system:
1. Fetches real-time data from the RDW Virtual Racing API
2. Processes the data through three different prediction models
3. Aggregates the predictions to create more reliable forecasts
4. Calculates optimal betting allocations
5. Places bets through the API

## Features

- **Data Collection**: Automatically fetches driver information, vehicle specifications, track details, and historical race results
- **Multiple Models**: Implements three different machine learning approaches:
  - Basic XGBoost classifier
  - Hyperparameter-tuned CatBoost model
  - Optuna-optimized XGBoost model
- **Ensemble Prediction**: Combines predictions from all three models to reduce variance
- **Adaptive Betting**: Dynamically decides whether to bet on top two or top three drivers based on mathematical calculations
- **Proportional Allocation**: Distributes betting amount according to each driver's win probability

## Requirements

- Python 3.12+
- Dependencies:
  - pandas
  - numpy
  - xgboost
  - catboost
  - optuna
  - scikit-learn
  - requests

## Usage

1. **Data Collection**:
   ```
   python vehicle.py
   ```
   This script fetches the latest data and creates three CSV files: `vehicles.csv`, `track.csv`, and `results.csv`.

2. **Run Predictions**:
   ```
   python runall.py
   ```
   This runs all three prediction models and generates an optimal betting strategy.

3. **Place Bets**:
   ```
   python main.py
   ```
   This script places the calculated bets through the RDW Virtual Racing API.

## How It Works

1. **Feature Engineering**: The system analyzes:
   - Vehicle data (weight, engine capacity, power-to-weight ratio)
   - Driver attributes (experience, driving style)
   - Track conditions (weather, time of day, track features)

2. **Model Training**: Each model is trained on historical race data to predict win probabilities.

3. **Ensemble Method**: The predictions from all three models are averaged to create a final prediction.

4. **Betting Logic**: The system calculates whether to bet on the top two or top three drivers based on expected value calculations.

5. **Bet Placement**: Bets are placed via the API, with funds distributed proportionally according to win probabilities.

## Project Structure

- `vehicle.py`: Fetches and processes data from the RDW API
- `test.py`: Implements basic XGBoost prediction model
- `test2.py`: Implements CatBoost model with GridSearchCV optimization
- `test3.py`: Implements XGBoost with Optuna hyperparameter optimization
- `runall.py`: Runs all prediction models and calculates optimal betting strategy
- `main.py`: Places bets through the RDW API

## Future Improvements

- Implement more sophisticated ensemble methods
- Add time-series analysis to account for driver/vehicle performance trends
- Incorporate more track-specific features
- Develop an automated system to continuously update and refine models


## Contributors

Sigurdur Haukur Birgisson
Josh van Vliet
Efe Özbal
Artur De Vlieger Lima
