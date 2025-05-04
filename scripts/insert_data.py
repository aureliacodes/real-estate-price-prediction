import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import sys
db_url = None
load_dotenv() 
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    db_url = DATABASE_URL 
else:
    db_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

df = pd.read_csv('./data/train.csv')

# Rename columns for compatibility
df = df.rename(columns={
    '1stFlrSF': 'FirstFlrSF',
    '2ndFlrSF': 'SecondFlrSF',
    '3SsnPorch': 'ThreeSeasonPorch',
    'BedroomAbvGr': 'Bedroom',
    'KitchenAbvGr': 'Kitchen',
})

# Replace NaN values with 0 
df = df.fillna({
    col: 0 if df[col].dtype in ['float64', 'int64'] else 'Unknown'
    for col in df.columns
})

# Create the conection engine to MySQL

engine = create_engine(db_url, connect_args={"connect_timeout": 15})
if engine.dialect.has_table(engine.connect(), 'properties'):
    print(" Table already exists — inserting rows.")
else:
    print("Table does not exist. Please create it before inserting.")


#Load directly into MySQL (if you want to add the data without deleting the table, use if_exists='append')
df.to_sql('properties', con=engine, if_exists='append', index=False)

print("Data inserted successfully with SQLAlchemy!")
