import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from model import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
airbnb_data = pd.read_csv('data/all_airbnb_processed.csv')
zillow_data = pd.read_csv('data/all_zillow_processed.csv')

# Airbnb feature specification
airbnb_numerical_features = ['accommodates', 'beds', 'minimum_nights', 'availability_365', 'number_of_reviews',
                             'review_scores_rating', 'review_scores_accuracy', 'review_scores_value', 'amenities_count']
airbnb_categorical_features = ['neighbourhood_cleansed', 'room_type', 'city']

# Zillow feature specification
zillow_numerical_features = [col for col in zillow_data.columns if col not in ['RegionName', 'RegionType', 'StateName', 'State', 'City', 'Metro', 'CountyName', '2023-12-31']]
zillow_categorical_features = ['RegionName', 'City']

# Load and split data
X_airbnb = airbnb_data.drop('price', axis=1)
y_airbnb = airbnb_data['price']

X_zillow = zillow_data.drop('2023-12-31', axis=1)
y_zillow = zillow_data['2023-12-31']

# Split the data for modeling
X_train_airbnb, X_temp_airbnb, y_train_airbnb, y_temp_airbnb = train_test_split(X_airbnb, y_airbnb, test_size=0.3, random_state=42)
X_val_airbnb, X_test_airbnb, y_val_airbnb, y_test_airbnb = train_test_split(X_temp_airbnb, y_temp_airbnb, test_size=0.5, random_state=42)

X_train_zillow, X_temp_zillow, y_train_zillow, y_temp_zillow = train_test_split(X_zillow, y_zillow, test_size=0.3, random_state=42)
X_val_zillow, X_test_zillow, y_val_zillow, y_test_zillow = train_test_split(X_temp_zillow, y_temp_zillow, test_size=0.5, random_state=42)

# Evaluate Airbnb models
print("Airbnb Models:")
for model_name in ["Linear Regression", "Random Forest", "XGBoost"]:
    base_model = load_model(f'models/{model_name}_base_airbnb.joblib')
    tuned_model = load_model(f'models/{model_name.lower()}_tuned_model_airbnb.joblib')

    print(f"\n{model_name} Base Model:")
    val_mae = mean_absolute_error(y_val_airbnb, base_model.predict(X_val_airbnb))
    val_r2 = r2_score(y_val_airbnb, base_model.predict(X_val_airbnb))
    test_mae = mean_absolute_error(y_test_airbnb, base_model.predict(X_test_airbnb))
    test_r2 = r2_score(y_test_airbnb, base_model.predict(X_test_airbnb))
    print(f"Validation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")
    print(f"Test MAE: {test_mae:.2f}, Test R-squared: {test_r2:.2f}")

    print(f"\n{model_name} Tuned Model:")
    val_mae = mean_absolute_error(y_val_airbnb, tuned_model.predict(X_val_airbnb))
    val_r2 = r2_score(y_val_airbnb, tuned_model.predict(X_val_airbnb))
    test_mae = mean_absolute_error(y_test_airbnb, tuned_model.predict(X_test_airbnb))
    test_r2 = r2_score(y_test_airbnb, tuned_model.predict(X_test_airbnb))
    print(f"Validation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")
    print(f"Test MAE: {test_mae:.2f}, Test R-squared: {test_r2:.2f}")

# Evaluate Zillow models
print("\nZillow Models:")
for model_name, is_keras in [("N-Beats", True), ("LSTM", True)]:
    base_model = load_model(f'models/{model_name.lower()}_base_model.h5', is_keras=True)
    tuned_model = load_model(f'models/{model_name.lower()}_tuned_model.h5', is_keras=True)

    print(f"\n{model_name} Base Model:")
    val_mae = mean_absolute_error(y_val_zillow, base_model.predict(X_val_zillow))
    val_r2 = r2_score(y_val_zillow, base_model.predict(X_val_zillow))
    test_mae = mean_absolute_error(y_test_zillow, base_model.predict(X_test_zillow))
    test_r2 = r2_score(y_test_zillow, base_model.predict(X_test_zillow))
    print(f"Validation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")
    print(f"Test MAE: {test_mae:.2f}, Test R-squared: {test_r2:.2f}")

    print(f"\n{model_name} Tuned Model:")
    val_mae = mean_absolute_error(y_val_zillow, tuned_model.predict(X_val_zillow))
    val_r2 = r2_score(y_val_zillow, tuned_model.predict(X_val_zillow))
    test_mae = mean_absolute_error(y_test_zillow, tuned_model.predict(X_test_zillow))
    test_r2 = r2_score(y_test_zillow, tuned_model.predict(X_test_zillow))
    print(f"Validation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")
    print(f"Test MAE: {test_mae:.2f}, Test R-squared: {test_r2:.2f}")

    # Visualizations for N-Beats model
    if model_name == "N-Beats":
        y_pred = tuned_model.predict(X_test_zillow)

        # Actual vs. Predicted Values Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test_zillow)), y_test_zillow, color='blue', label='Actual', alpha=0.5)
        plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.5)
        plt.title('Actual vs Predicted Values (N-Beats Model)')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

        # Check and reshape if necessary
        print("Shape of y_test:", y_test_zillow.shape)
        print("Shape of y_pred:", y_pred.shape)

        if len(y_test_zillow.shape) > 1:
            y_test_zillow = y_test_zillow.reshape(-1)
        if len(y_pred.shape) > 1:
            y_pred = y_pred.reshape(-1)

        # Calculating Residuals
        residuals = y_test_zillow - y_pred

        # Residuals Plot
        plt.figure(figsize=(10, 6))
        sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
        plt.title('Residuals vs Predicted (N-Beats Model)')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.axhline(0, linestyle='--', color='gray')
        plt.show()

