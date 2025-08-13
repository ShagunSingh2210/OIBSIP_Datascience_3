# OIBSIP_Datascience_3
3. Car Price Prediction

Objective
Predict selling prices of used cars based on features like present price, kilometers driven, fuel type, transmission, ownership, and age.

Steps
Loaded and cleaned car data.csv.
Created Car_Age from Year, dropped unnecessary columns.
Scaled numerical data, one-hot encoded categorical data.
Trained Random Forest & XGBoost models.
Used Random Forest with GridSearchCV for best parameters.
Saved final model as car_price_predictor.pkl.
Added function to predict price for new car details.

Tools
Python, pandas, numpy, scikit-learn, xgboost, joblib.

Outcome
Random Forest gave low RMSE and high R², making accurate price predictions possible.
