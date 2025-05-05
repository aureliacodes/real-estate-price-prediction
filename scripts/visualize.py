"""Train and evaluate a linear regression model for house price prediction."""

import os
import pickle
from dotenv import load_dotenv

import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load environment variables
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")
DATABASE_URL = os.getenv("DATABASE_URL")

# Compose database URL
if DATABASE_URL:
    db_url = DATABASE_URL
else:
    db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Connect to the database
engine = create_engine(db_url, connect_args={"connect_timeout": 15})

# Read training data
df = pd.read_sql(
    """
    SELECT OverallQual, GrLivArea, GarageCars, GarageArea, YearBuilt, SalePrice
    FROM properties
    WHERE SalePrice IS NOT NULL
    """,
    engine,
)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Prepare features and target
features = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "YearBuilt"]
X = df[features]
y = df["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model trained. Mean Squared Error on test set: {mse:.2f}")

# Predict on new test data
TEST_FILE = "data/test.csv"

if os.path.exists(TEST_FILE):
    test_df = pd.read_csv(TEST_FILE)

    if all(col in test_df.columns for col in features):
        test_df.fillna(df.median(numeric_only=True), inplace=True)
        X_new = test_df[features]
        test_df["predicted_price"] = model.predict(X_new)

        # Save predictions
        OUTPUT_FILE = "data/predictions.csv"
        test_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Predictions saved to '{OUTPUT_FILE}'.")

    else:
        missing_cols = list(set(features) - set(test_df.columns))
        print(f"Missing required columns in 'test.csv': {missing_cols}")
else:
    print("File 'data/test.csv' not found.")

# Save the model
os.makedirs("model", exist_ok=True)
with open("model/linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'model/linear_model.pkl'")
