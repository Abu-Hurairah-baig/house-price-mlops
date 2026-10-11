# predict.py - Use the saved model to predict the price of a new house

import pandas as pd
import joblib

# Step 1: Load the trained model saved by train.py
model = joblib.load("model.pkl")

# Step 2: Describe the new house (column names must match training)
new_house = pd.DataFrame({"rooms": [3], "size_sqft": [1350]})

# Step 3: Predict the price
price = model.predict(new_house)[0]

# Step 4: Show the result
print(f"3 rooms, 1350 sqft -> predicted price: {price:.1f} lakh")
