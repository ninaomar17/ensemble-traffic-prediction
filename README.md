# ensemble-traffic-prediction

## Overview
This project focuses on predicting traffic volume using a combination of feature selection techniques and machine learning models, specifically Gradient Boosting (GBM) and Long Short-Term Memory (LSTM) networks. The final predictions are obtained through an ensemble method that combines both models.

## Features
- **Feature Selection**: Correlation-based filtering and Recursive Feature Elimination (RFE) using Gradient Boosting.
- **Machine Learning Models**:
  - **Gradient Boosting Classifier (GBM)** for structured data classification.
  - **LSTM Model** for time series prediction.
- **Ensemble Learning**: Weighted voting approach to combine predictions from both models.
- **Hyperparameter Tuning**: Grid Search for GBM and Random Search with Keras Tuner for LSTM.
- **Performance Evaluation**: Accuracy, precision, recall, F1-score, and classification report.

## Data
The dataset used in this project is stored in `TrafficVolumeData.csv`. It includes various features such as:
- `date_time`: Timestamp of traffic volume measurement.
- `traffic_volume`: The target variable representing the number of vehicles.
- `is_holiday`, `weather_type`, `weather_description`: Additional contextual information.

## Preprocessing Steps
1. **Handling Missing Values**: Replaces NaN values with `0`.
2. **DateTime Feature Engineering**:
   - Extracts `hour`, `day_of_week`, and `month` from `date_time`.
3. **Encoding Categorical Features**:
   - `is_holiday` is converted to binary values.
   - `weather_type` and `weather_description` are label-encoded.
4. **Binning Target Variable**: `traffic_volume` is categorized into `low`, `medium`, and `high` using quantiles.
5. **Feature Selection**:
   - Correlation-based filtering to remove highly correlated features.
   - Recursive Feature Elimination (RFE) using GBM to select the best predictors.
6. **Data Normalization**: Min-Max Scaling is applied to selected features.

## Model Training
### 1. Gradient Boosting Classifier (GBM)
- The model is trained on selected features after feature engineering.
- Hyperparameters: `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`.

### 2. LSTM Model
- Uses time series windowing with a `window_size` of 24.
- Hyperparameter tuning is performed using Keras Tuner (Random Search).
- Best model is saved and fine-tuned for 20 epochs.

### 3. Ensemble Learning
- Weighted voting is applied with equal weights for GBM and LSTM to obtain final predictions.

## Evaluation Metrics
The performance of each model and the ensemble approach is assessed using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Classification Report**

## Installation & Dependencies
Ensure you have the following Python packages installed:
```bash
pip install pandas numpy scikit-learn tensorflow keras-tuner
```

## Running the Code
1. Place `TrafficVolumeData.csv` in the specified directory.
2. Run the script to preprocess data, train models, and evaluate results.
```bash
python traffic_prediction.py
```

## Results
The ensemble model aims to improve prediction accuracy by leveraging the strengths of both GBM and LSTM, ensuring robustness in traffic volume forecasting.


