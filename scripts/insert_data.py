"""
insert_data.py - Script to insert CSV data into a MySQL database using SQLAlchemy.
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "3306")
DATABASE_URL = os.getenv("DATABASE_URL")

# Build the database connection URL
if DATABASE_URL:
    database_url = DATABASE_URL
else:
    database_url = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

# Load the dataset
df = pd.read_csv("./data/train.csv")

# Rename columns for compatibility
df = df.rename(
    columns={
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "ThreeSeasonPorch",
        "BedroomAbvGr": "Bedroom",
        "KitchenAbvGr": "Kitchen",
    }
)

# Replace NaN values with 0 or 'Unknown' depending on type
df = df.fillna(
    {
        col: 0 if df[col].dtype in ["float64", "int64"] else "Unknown"
        for col in df.columns
    }
)

# Drop ID column
df = df.drop(columns=["Id"])

# Create SQLAlchemy engine
engine = create_engine(database_url, connect_args={"connect_timeout": 15})

# Check if the table exists
with engine.connect() as connection:
    if engine.dialect.has_table(connection, "properties"):
        print("Table already exists — inserting rows.")
    else:
        print("Table does not exist. Please create it before inserting.")

# Insert data
df.to_sql("properties", con=engine, if_exists="append", index=False)
print("Data inserted successfully with SQLAlchemy!")
