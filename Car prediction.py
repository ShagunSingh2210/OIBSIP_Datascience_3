import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

# --- 1. Data Loading and Preprocessing ---
# FIX: The file path was corrected to the full path to avoid FileNotFoundError.
# Please ensure this path matches the location of your 'car data.csv' file.
file_path = r'C:\Users\Shagun Singh\Desktop\Python\car data.csv'
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()

# Engineer a new feature for car age and drop the original 'Year' column
df['Car_Age'] = 2023 - df['Year']
df.drop('Year', axis=1, inplace=True)

# Separate features (X) and target (y)
X = df.drop(['Selling_Price', 'Car_Name'], axis=1)
y = df['Selling_Price']

# Identify categorical and numerical columns for preprocessing
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

# Create a preprocessor to handle numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Model Training and Evaluation ---
print("\n--- Training and Evaluating Initial Models ---")
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting (XGBoost)': xgb.XGBRegressor(random_state=42),
}

results = {}
for name, model in models.items():
    # Create a pipeline to streamline preprocessing and modeling
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions and evaluate
    preds = pipeline.predict(X_test)
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds)
    }

    print(f"\n{name} Results:")
    print(f"  RMSE: {results[name]['RMSE']:.2f}")
    print(f"  R2 Score: {results[name]['R2']:.2f}")

# --- 3. Hyperparameter Tuning for Random Forest ---
print("\n--- Hyperparameter Tuning for Random Forest ---")
# Create a new pipeline for the Random Forest model to tune its parameters
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))])

# Define the grid of parameters to search
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5, 10]
}

# Perform a grid search with cross-validation to find the best parameters
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Evaluate the best model on the test set
final_preds = best_rf_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
final_r2 = r2_score(y_test, final_preds)

print(f"\nFinal Model (Random Forest with best parameters) Results:")
print(f"  RMSE: {final_rmse:.2f}")
print(f"  R2 Score: {final_r2:.2f}")

# --- 4. Saving and Using the Final Model ---
# Save the best model for future use
joblib.dump(best_rf_model, 'car_price_predictor.pkl')
print("\nBest model saved as 'car_price_predictor.pkl'")

# Define a function to make a single prediction
def predict_car_price(model, present_price, driven_kms, fuel_type, selling_type, transmission, owner, car_age):
    """
    Predicts the selling price of a car given its features.
    The input data must be a pandas DataFrame with the same column names as the training data.
    """
    data = pd.DataFrame({
        'Present_Price': [present_price],
        'Driven_kms': [driven_kms],
        'Fuel_Type': [fuel_type],
        'Selling_type': [selling_type],
        'Transmission': [transmission],
        'Owner': [owner],
        'Car_Age': [car_age]
    })
    return model.predict(data)[0]

# Example of how to use the saved model
print("\n--- Example Prediction ---")
# You can load the model back from the file if needed
# loaded_model = joblib.load('car_price_predictor.pkl')
predicted_price = predict_car_price(best_rf_model, 
                                    present_price=8.5, 
                                    driven_kms=25000, 
                                    fuel_type='Petrol', 
                                    selling_type='Dealer', 
                                    transmission='Manual', 
                                    owner=0, 
                                    car_age=5)
print(f"The predicted selling price for the example car is: {predicted_price:.2f} Lakhs")