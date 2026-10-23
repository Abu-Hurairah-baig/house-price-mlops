# train.py - A beginner machine learning script: predict house prices
# Usage: python train.py [linear|tree|forest]   (default is linear)

import sys
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# MLflow setup: store the experiment records in a local database file
# and group all our runs under one experiment name
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("house-price-project")

# Choose the model type from the command line (default: linear)
# sys.argv is the list of words typed after "python train.py"
model_type = sys.argv[1] if len(sys.argv) > 1 else "linear"
if model_type not in ["linear", "tree", "forest"]:
    print("Unknown model type:", model_type)
    print("Please choose one of: linear, tree, forest")
    sys.exit(1)

# Step 1: Read the data from houses.csv
data = pd.read_csv("houses.csv")

# Step 2: Choose features (X) and the label (y)
# Features are the inputs; the label is what we want to predict
X = data[["rooms", "size_sqft"]]
y = data["price_lakh"]

# Step 3: Split into training data (80%) and testing data (20%)
# random_state=42 makes the split the same every time
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
print("Houses used for training:", len(X_train))
print("Houses used for testing:", len(X_test))

# Step 4: Start an MLflow run - everything inside the "with" block
# (training and scoring) is recorded as one run
with mlflow.start_run():

    # Step 5: Create the model chosen on the command line, then train it
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "tree":
        model = DecisionTreeRegressor(random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    # Step 6: Predict prices for the test houses and compare with real prices
    predictions = model.predict(X_test)

    print("\nTest results:")
    for i in range(len(X_test)):
        rooms = X_test.iloc[i]["rooms"]
        size = X_test.iloc[i]["size_sqft"]
        real = y_test.iloc[i]
        guess = predictions[i]
        off_by = abs(real - guess)
        print(f"Rooms: {rooms}, Size: {size} sqft, Real price: {real}, "
              f"Guess: {guess:.1f}, Off by: {off_by:.1f}")

    # Step 7: Calculate the Mean Absolute Error (average size of the mistakes)
    mae = mean_absolute_error(y_test, predictions)
    print(f"\nModel type: {model_type} | Mean Absolute Error (MAE): {mae:.2f} lakh")

    # Step 8: Log to MLflow: the settings (parameters) and the score (metric)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("rows_in_dataset", len(data))
    mlflow.log_metric("mae", mae)

    # Save the score to a file so the CI quality gate can read it
    with open("metrics.json", "w") as f:
        json.dump({"mae": mae}, f)

    # Step 9: Log the trained model to MLflow and register it by name
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="house-price-model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

# Step 10: Also save the trained model to a file so we can use it later
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")