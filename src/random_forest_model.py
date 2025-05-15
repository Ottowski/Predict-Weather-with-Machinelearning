from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

def train_random_forest(X_train, X_test, y_train, y_test):
    print("Training Random Forest model on training data...")
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    print("Model training complete.")

    print("Making predictions on test data...")
    y_pred = rf_model.predict(X_test)

    print("Calculating evaluation metrics...")
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")

    return rf_model
