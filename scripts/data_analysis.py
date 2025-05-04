import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

load_dotenv() 
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    db_url = DATABASE_URL 
else:
    db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
query = "SELECT * FROM properties"

try:
    # Establish connection and read data
    engine = create_engine(db_url)
    df = pd.read_sql(query, engine)

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
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    print("\nMissing Values Analysis:")
    print(pd.DataFrame({'Missing Count': missing_values, 'Missing Percent': missing_percent}))

    # Unique values in categorical columns
    def analyze_unique_values(df, categorical_columns):
        for col in categorical_columns:
            print(f"\nUnique values in {col}:")
            print(df[col].value_counts())

    analyze_unique_values(df, df.select_dtypes(include=['object']).columns)

    # Correlation analysis
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    corr_filtered = correlation_matrix.where(
        (correlation_matrix.abs() > 0.5) & (correlation_matrix.abs() < 1.0)
    ).dropna(how='all').dropna(axis=1, how='all')
    print("\nCorrelation Analysis (0.5 < |r| < 1.0):")
    print(corr_filtered)

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
  

    # Use interquartile range (IQR) to detect outliers in numeric columns
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        outliers = detect_outliers(df, col)
        print(f"\nOutliers in {col}: {len(outliers)}")

    # Verify required columns exist
    required_cols = ['SalePrice', 'GrLivArea', 'YrSold', 'YearBuilt', 'Neighborhood', 'MoSold']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # --- Derived Metrics
    df['price_per_sqm'] = df['SalePrice'] / df['GrLivArea']
    print("\nPrice per Square Meter:")
    print(df[['SalePrice', 'GrLivArea', 'price_per_sqm']].head())

    df['property_age'] = df['YrSold'] - df['YearBuilt']
    print("\nAge of Property:")
    print(df[['YearBuilt', 'YrSold', 'property_age']].head())

    # ----- Aggregations
    grouped_data = numeric_df.groupby('Neighborhood').agg({
        'SalePrice': ['mean', 'max', 'min'],
        'GrLivArea': 'mean'
    }).sort_values(('SalePrice', 'mean'), ascending=False)

    print("\nGrouped Analysis by Neighborhood:")
    print(grouped_data)

    # --- Trend Analysis
    monthly_sales = df.groupby('MoSold')['SalePrice'].mean()
    print("\nAverage SalePrice by Month:")
    print(monthly_sales)

    yearly_sales = df.groupby('YrSold')['SalePrice'].mean()
    print("\nAverage SalePrice by Year:")   
    print(yearly_sales)

    # Data Completeness Check
    complete_rows = df.dropna()
    print("\nNumber of Complete Rows:", len(complete_rows))

    # Duplicate Data Check
    duplicates = df[df.duplicated()]
    print("\nDuplicate Rows:", len(duplicates))
    print(duplicates)

except Exception as e:
    print(f"Error during script execution: {e}")
