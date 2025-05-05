"""
Script to create a MySQL database and table for real estate data.
"""

import os

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME", "real_estate")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DATABASE_URL = os.getenv("DATABASE_URL")

# Build database URLs: one without the DB, one with the DB
if DATABASE_URL:
    db_url_no_db = DATABASE_URL.rsplit("/", 1)[0]
else:
    db_url_no_db = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"

DB_URL_WITH_DB = f"{db_url_no_db}/{DB_NAME}"


try:
    # Step 1: Connect to MySQL server and create database if it doesn't exist
    engine = create_engine(db_url_no_db, connect_args={"connect_timeout": 15})
    with engine.connect() as connection:
        print("Connected to MySQL server successfully.")
        print(f"Creating database '{DB_NAME}' if it doesn't exist...")
        connection.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS {DB_NAME} DEFAULT CHARACTER SET utf8mb4;"
            )
        )

    # Step 2: Connect to the 'real_estate' database and create the table
    engine_real_estate = create_engine(
        DB_URL_WITH_DB, connect_args={"connect_timeout": 15}
    )
    with engine_real_estate.connect() as connection:
        print(f"Connected to database '{DB_NAME}' successfully.")
        print("Creating table 'properties' if it doesn't exist...")
        connection.execute(
            text(
                """
CREATE TABLE IF NOT EXISTS properties (
    id INT AUTO_INCREMENT PRIMARY KEY,
    MSSubClass INT,
    MSZoning VARCHAR(255),
    LotFrontage FLOAT,
    LotArea FLOAT,
    Street VARCHAR(255),
    Alley VARCHAR(255),
    LotShape VARCHAR(255),
    LandContour VARCHAR(255),
    Utilities VARCHAR(255),
    LotConfig VARCHAR(255),
    LandSlope VARCHAR(255),
    Neighborhood VARCHAR(255),
    Condition1 VARCHAR(255),
    Condition2 VARCHAR(255),
    BldgType VARCHAR(255),
    HouseStyle VARCHAR(255),
    OverallQual INT,
    OverallCond INT,
    YearBuilt INT,
    YearRemodAdd INT,
    RoofStyle VARCHAR(255),
    RoofMatl VARCHAR(255),
    Exterior1st VARCHAR(255),
    Exterior2nd VARCHAR(255),
    MasVnrType VARCHAR(255),
    MasVnrArea FLOAT,
    ExterQual VARCHAR(255),
    ExterCond VARCHAR(255),
    Foundation VARCHAR(255),
    BsmtQual VARCHAR(255),
    BsmtCond VARCHAR(255),
    BsmtExposure VARCHAR(255),
    BsmtFinType1 VARCHAR(255),
    BsmtFinSF1 FLOAT,
    BsmtFinType2 VARCHAR(255),
    BsmtFinSF2 FLOAT,
    BsmtUnfSF FLOAT,
    TotalBsmtSF FLOAT,
    Heating VARCHAR(255),
    HeatingQC VARCHAR(255),
    CentralAir VARCHAR(255),
    Electrical VARCHAR(255),
    FirstFlrSF FLOAT,
    SecondFlrSF FLOAT,
    LowQualFinSF FLOAT,
    GrLivArea FLOAT,
    BsmtFullBath INT,
    BsmtHalfBath INT,
    FullBath INT,
    HalfBath INT,
    Bedroom INT,
    Kitchen INT,
    KitchenQual VARCHAR(255),
    TotRmsAbvGrd INT,
    Functional VARCHAR(255),
    Fireplaces INT,
    FireplaceQu VARCHAR(255),
    GarageType VARCHAR(255),
    GarageYrBlt INT,
    GarageFinish VARCHAR(255),
    GarageCars INT,
    GarageArea FLOAT,
    GarageQual VARCHAR(255),
    GarageCond VARCHAR(255),
    PavedDrive VARCHAR(255),
    WoodDeckSF FLOAT,
    OpenPorchSF FLOAT,
    EnclosedPorch FLOAT,
    ThreeSeasonPorch FLOAT,
    ScreenPorch FLOAT,
    PoolArea FLOAT,
    PoolQC VARCHAR(255),
    Fence VARCHAR(255),
    MiscFeature VARCHAR(255),
    MiscVal FLOAT,
    MoSold INT,
    YrSold INT,
    SaleType VARCHAR(255),
    SaleCondition VARCHAR(255),
    SalePrice FLOAT
);
        """
            )
        )
        print("Table 'properties' created or already exists.")

except SQLAlchemyError as err:
    print(f"!!! SQLAlchemy Error: {err}")
except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"!!! Unexpected error: {type(e).__name__}: {e}")

finally:
    print("Script finished successfully.")
