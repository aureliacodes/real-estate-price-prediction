"""Train a linear regression model and generate predictions for house prices."""

import os
import pickle
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Constante
FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "YearBuilt"]
TEST_FILE = "data/test.csv"
OUTPUT_FILE = "data/predictions.csv"
MODEL_PATH = "model/linear_model.pkl"

# Load .env
load_dotenv()

def get_database_url():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    else:
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_NAME = os.getenv("DB_NAME")
        DB_PORT = os.getenv("DB_PORT")
        return f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def load_training_data(engine):
    """Load training data from the MySQL database."""
    df = pd.read_sql(
        """
        SELECT OverallQual, GrLivArea, GarageCars, GarageArea, YearBuilt, SalePrice
        FROM properties
        WHERE SalePrice IS NOT NULL
        """,
        engine,
    )
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def train_model(X, y):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance using Mean Squared Error."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained. Mean Squared Error on test set: {mse:.2f}")
    return mse

def save_model(model, path=MODEL_PATH):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to '{path}'.")

def predict_and_save(model, test_file=TEST_FILE, output_file=OUTPUT_FILE):
    """Predict using test data and save the predictions."""
    if not os.path.exists(test_file):
        print(f"Test file '{test_file}' not found.")
        return
    
    test_df = pd.read_csv(test_file)

    if not all(col in test_df.columns for col in FEATURES):
        missing_cols = list(set(FEATURES) - set(test_df.columns))
        print(f"Missing required columns in '{test_file}': {missing_cols}")
        return

    test_df.fillna(test_df.median(numeric_only=True), inplace=True)
    test_df["predicted_price"] = model.predict(test_df[FEATURES])
    test_df.to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'.")

def main():
    """Main pipeline for training and predicting."""
    db_url = get_database_url()
    engine = create_engine(db_url, connect_args={"connect_timeout": 15})
    
    # Load data and train model
    df = load_training_data(engine)
    X = df[FEATURES]
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    predict_and_save(model)

if __name__ == "__main__":
    main()
