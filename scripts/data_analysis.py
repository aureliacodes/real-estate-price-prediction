"""
Data Analysis Script
- This script connects to a MySQL database, retrieves data from the 'properties' table,
  and performs various data analysis tasks including descriptive statistics,
  correlation analysis, outlier detection, and trend analysis.
- It also checks for missing values, unique values in categorical columns, and duplicates.
"""

# Standard library imports
import os

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define constants
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "3306")  # Default as string
DATABASE_URL = os.getenv("DATABASE_URL")
QUERY = "SELECT * FROM properties"

# Define database connection string
DB_URL = (DATABASE_URL if DATABASE_URL
           else f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

try:
    # Establish connection and read data
    engine = create_engine(DB_URL)
    df = pd.read_sql(QUERY, engine)

    # Display DataFrame preview
    print("First 5 rows:")
    print(df.head())

    # General information about dataset
    print("\nInfo about dataset:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.info())

    # Summary statistics for numeric columns
    print("\nSummary statistics:")
    print(df.describe())

    # Missing values analysis
    def analyze_missing_values(dataframe):
        """Analyzes missing values in the dataset."""
        missing_values = dataframe.isnull().sum()
        missing_percent = (missing_values / len(dataframe)) * 100
        return pd.DataFrame({"Missing Count": missing_values, "Missing Percent": missing_percent})

    print("\nMissing Values Analysis:")
    print(analyze_missing_values(df))

    # Unique values in categorical columns
    def analyze_unique_values(dataframe, categorical_columns):
        """Prints unique values count for categorical columns."""
        for category in categorical_columns:
            print(f"\nUnique values in {category}:")
            print(dataframe[category].value_counts())

    analyze_unique_values(df, df.select_dtypes(include=["object"]).columns)

    # Correlation analysis
    numeric_df = df.select_dtypes(include=["number"])
    correlation_matrix = numeric_df.corr()
    corr_filtered = correlation_matrix.where(
        (correlation_matrix.abs() > 0.5) & (correlation_matrix.abs() < 1.0)
    ).dropna(how="all").dropna(axis=1, how="all")

    print("\nCorrelation Analysis (0.5 < |r| < 1.0):")
    print(corr_filtered)

    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")

    # Outlier detection using IQR method
    def detect_outliers(dataframe, column):
        """Detects outliers in a numeric column using IQR."""
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        outliers = detect_outliers(df, col)
        print(f"\nOutliers in {col}: {len(outliers)}")

    # Verify required columns exist
    required_cols = ["SalePrice", "GrLivArea", "YrSold", "YearBuilt", "Neighborhood", "MoSold"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Derived Metrics
    df["price_per_sqm"] = df["SalePrice"] / df["GrLivArea"]
    print("\nPrice per Square Meter:")
    print(df[["SalePrice", "GrLivArea", "price_per_sqm"]].head())

    df["property_age"] = df["YrSold"] - df["YearBuilt"]
    print("\nAge of Property:")
    print(df[["YearBuilt", "YrSold", "property_age"]].head())

    # Aggregations
    grouped_data = numeric_df.groupby("Neighborhood").agg(
        {"SalePrice": ["mean", "max", "min"], "GrLivArea": "mean"}
    ).sort_values(("SalePrice", "mean"), ascending=False)

    print("\nGrouped Analysis by Neighborhood:")
    print(grouped_data)

    # Trend Analysis
    monthly_sales = df.groupby("MoSold")["SalePrice"].mean()
    print("\nAverage SalePrice by Month:")
    print(monthly_sales)

    yearly_sales = df.groupby("YrSold")["SalePrice"].mean()
    print("\nAverage SalePrice by Year:")
    print(yearly_sales)

    # Data Completeness Check
    complete_rows = df.dropna()
    print("\nNumber of Complete Rows:", len(complete_rows))

    # Duplicate Data Check
    duplicates = df[df.duplicated()]
    print("\nDuplicate Rows:", len(duplicates))
    print(duplicates)
except ValueError as err:
    print(f"ValueError: {err}")
except KeyError as err:
    print(f"KeyError: {err}")
except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Unexpected error: {type(e).__name__}: {e}")
