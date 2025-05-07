import os
import pickle
import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scripts import ml_model

# Fixtures

@pytest.fixture
def sample_data():
    """Generate sample data for training and testing."""
    df = pd.DataFrame({
        "OverallQual": [5, 6, 7, 8, 9],
        "GrLivArea": [1500, 1600, 1700, 1800, 1900],
        "GarageCars": [2, 2, 3, 3, 4],
        "GarageArea": [400, 420, 440, 460, 480],
        "YearBuilt": [2000, 2001, 2002, 2003, 2004],
        "SalePrice": [200000, 210000, 250000, 270000, 300000]
    })
    return df

def test_train_model(sample_data):
    """Test if model is trained and coefficients are not None."""
    X = sample_data[ml_model.FEATURES]
    y = sample_data["SalePrice"]
    model = ml_model.train_model(X, y)
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")

def test_evaluate_model(sample_data):
    """Test the MSE evaluation function returns a float >= 0."""
    X = sample_data[ml_model.FEATURES]
    y = sample_data["SalePrice"]
    X_train, X_test, y_train, y_test = ml_model.train_test_split(X, y, test_size=0.2, random_state=42)
    model = ml_model.train_model(X_train, y_train)
    mse = ml_model.evaluate_model(model, X_test, y_test)
    assert isinstance(mse, float)
    assert mse >= 0

def test_save_and_load_model(tmp_path, sample_data):
    """Test saving and loading a trained model using pickle."""
    model = ml_model.train_model(sample_data[ml_model.FEATURES], sample_data["SalePrice"])
    model_path = tmp_path / "test_model.pkl"
    ml_model.save_model(model, path=model_path)
    
    assert model_path.exists()
    
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    assert isinstance(loaded_model, LinearRegression)
    assert np.allclose(model.coef_, loaded_model.coef_)

def test_predict_and_save(tmp_path, sample_data):
    """Test prediction generation and CSV output."""
    model = ml_model.train_model(sample_data[ml_model.FEATURES], sample_data["SalePrice"])
    
    test_file = tmp_path / "test.csv"
    output_file = tmp_path / "predictions.csv"
    
    sample_data[ml_model.FEATURES].to_csv(test_file, index=False)
    
    ml_model.predict_and_save(model, test_file=test_file, output_file=output_file)
    
    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert "predicted_price" in df.columns
    assert not df["predicted_price"].isnull().any()
