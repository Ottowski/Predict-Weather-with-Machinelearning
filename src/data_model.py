from sklearn.linear_model import LinearRegression

def train_linear_regression(WH):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error

    features = ["avg_temp", "humidity", "wind_speed", "pressure"]
    X = WH[features]
    y = WH['avg_temp']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train) # Trains model

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}")

    return model
