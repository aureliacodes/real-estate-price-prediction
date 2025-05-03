
# 🏠 Real Estate Price Prediction

This project showcases a complete **Data Analysis** and **Machine Learning** pipeline for predicting house prices using real estate data. It demonstrates data handling from CSV to SQL database, model training, evaluation, and prediction on unseen data.

---

## ✅ Project Highlights

- **Data Pipeline:** Imports Kaggle data (`train.csv`, `test.csv`) into a MySQL database.
- **Data Analysis:** Cleans and explores the dataset using Pandas.
- **Model Training & Evaluation:** Builds, trains (Linear Regression), and evaluates (MSE) a price prediction model.
- **Prediction Generation:** Predicts prices on unseen data (`test.csv`) and exports results to `data/predictions.csv`.

---

## 🧰 Technologies Used

- **Python 3**
- **MySQL**
- **SQLAlchemy** (for database connection)
- **Pandas**
- **Scikit-Learn**
- **Matplotlib**

---
## ⚠️ Environment Variables
This project uses environment variables. Create a `.env` file with the following content:

```env
DB_USER=your_db_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_NAME=real_estate


## 🚀 Getting Started

To run this project on your local machine:

1. Clone the repository  
2. Install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```

3. Create the MySQL database:  
   ```bash
   python scripts/create_db.py
   ```

4. Insert training data into the database:  
   ```bash
   python scripts/insert_data.py
   ```

5. Run data analysis (optional):  
   ```bash
   python scripts/data_analysis.py
   ```

6. Train the model and generate predictions:  
   ```bash
   python scripts/ml_model.py
   ```

7. (Optional) Visualize key insights:  
   ```bash
   python scripts/visualize.py
   ```

---

## 📊 Dataset Description

The dataset comes from [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). It includes:

- `OverallQual`: Overall material and finish quality
- `GrLivArea`: Above ground living area in square feet
- `GarageCars`: Size of garage in car capacity
- `GarageArea`: Size of garage in square feet
- `YearBuilt`: Original construction date
- `SalePrice`: Property sale price (only in training data)

The target is to **predict the SalePrice** of each house using the other features.

---

## 🤖 Machine Learning Model

- **Linear Regression** is used as the base model.
- Data is split into training and test sets (80/20).
- Missing values are handled using the median strategy.
- Model evaluation is done using **Mean Squared Error (MSE)**.
- Predictions are made on `test.csv` and saved in `data/predictions.csv`.

---

## 📁 Output

After running the full pipeline:
- Model is trained and tested
- MSE is printed in the terminal
- Predictions are saved here:  
  ```
  data/predictions.csv
  ```
  model/linear_model.pkl
---


## 👩‍💻 Author

**Aurelia Cucereavii**  

---

📬 Feedback & Contributions
Suggestions, bug reports, or questions? Please **open an issue** on this repository. Contributions are welcome!


📝 License
This project is licensed under the MIT License, meaning you are free to use, modify, and share it.

Thank you for your time and interest! I hope this project is useful to you, and I'm excited to collaborate with passionate tech enthusiasts. 🚀