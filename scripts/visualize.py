import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import os


def validate_columns(df, columns_list, auto_convert=False):
    """
    Validate that specified columns exist and are numeric.
    Optionally attempt to convert non-numeric columns to numeric.
    """
    missing_columns = [col for col in columns_list if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False

    non_numeric_columns = [col for col in columns_list if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_columns:
        if auto_convert:
            for col in non_numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            still_non_numeric = [col for col in non_numeric_columns if not pd.api.types.is_numeric_dtype(df[col])]
            if still_non_numeric:
                print(f"Could not convert to numeric: {still_non_numeric}")
                return False
            else:
                print(f"Successfully converted {non_numeric_columns} to numeric.")
                return True
        else:
            print(f"Non-numeric columns: {non_numeric_columns}")
            return False
    return True


if __name__ == "__main__":
    # Read the dataset
    train_df = pd.read_csv('data/train.csv')

    if 'SalePrice' not in train_df.columns:
        print("SalePrice column is missing. Creating dummy data.")
        train_df['SalePrice'] = train_df['LotArea'] * 100
    else:
        print("SalePrice column found.")

    train_df = train_df.dropna(subset=['SalePrice', 'LotArea'])
    train_df = train_df[train_df['LotArea'] > 0]  # Prevent divide-by-zero

    if train_df.empty:
        print("No data available for analysis.")
    else:
        os.makedirs('visualizations', exist_ok=True)

        # Histogram: SalePrice
        plt.figure(figsize=(10, 6))
        plt.hist(train_df['SalePrice'], bins=30, color='blue', edgecolor='black', alpha=0.7)
        plt.xlabel('SalePrice')
        plt.ylabel('Frequency')
        plt.title('Distribution of SalePrice')
        plt.grid(True)
        plt.savefig('visualizations/saleprice_distribution.png')
        plt.close()

        # Correlation Heatmap (top 10 features)
        numeric_df = train_df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr()
        top_corr = correlation_matrix['SalePrice'].abs().sort_values(ascending=False).head(10).index
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix.loc[top_corr, top_corr], annot=True, cmap="coolwarm", fmt=".2f")
        plt.title('Top Correlations with SalePrice')
        plt.savefig('visualizations/correlation_heatmap.png')
        plt.close()

        # LotArea vs SalePrice with trendline
        if validate_columns(train_df, ['LotArea', 'SalePrice'], auto_convert=True):
            slope, intercept, _, _, _ = linregress(train_df['LotArea'], train_df['SalePrice'])
            trendline = slope * train_df['LotArea'] + intercept

            plt.figure(figsize=(10, 6))
            plt.scatter(train_df['LotArea'], train_df['SalePrice'], alpha=0.5, label='Data')
            plt.plot(train_df['LotArea'], trendline, color='red', label='Trend Line')
            plt.xlabel('Area (m²)')
            plt.ylabel('SalePrice')
            plt.title('Price vs Area with Trend Line')
            plt.legend()
            plt.grid(True)
            plt.savefig('visualizations/price_vs_area_trendline.png')
            plt.close()
        else:
            print("Cannot plot Price vs Area. Required columns are missing or not numeric.")

        # Bar chart: Average SalePrice by Neighborhood
        neighborhood_prices = train_df.groupby('Neighborhood')['SalePrice'].mean().sort_values()
        plt.figure(figsize=(12, 8))
        neighborhood_prices.plot(kind='barh', color='orange', edgecolor='black')
        plt.xlabel('Average SalePrice')
        plt.ylabel('Neighborhood')
        plt.title('Average SalePrice by Neighborhood')
        plt.grid(True)
        plt.savefig('visualizations/neighborhood_prices.png')
        plt.close()

        # Histogram: Price per square meter
        train_df['price_per_sqm'] = train_df['SalePrice'] / train_df['LotArea']
        plt.figure(figsize=(10, 6))
        plt.hist(train_df['price_per_sqm'], bins=30, color='green', edgecolor='black', alpha=0.7)
        plt.xlabel('Price per Square Meter')
        plt.ylabel('Frequency')
        plt.title('Distribution of Price per Square Meter')
        plt.grid(True)
        plt.savefig('visualizations/price_per_sqm.png')
        plt.close()

        # Boxplot: SalePrice
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=train_df['SalePrice'], color='purple', showfliers=True)
        plt.title('Boxplot of SalePrice')
        plt.grid(True)
        plt.savefig('visualizations/saleprice_boxplot.png')
        plt.close()

        # Bar chart: Monthly Sales Trends
        monthly_sales = train_df.groupby('MoSold')['SalePrice'].mean()
        plt.figure(figsize=(10, 6))
        monthly_sales.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.xlabel('Month')
        plt.ylabel('Average SalePrice')
        plt.title('Monthly Sales Trends')
        plt.grid(True)
        plt.savefig('visualizations/monthly_sales_trends.png')
        plt.close()

        # Line chart: Yearly Sales Trends
        yearly_sales = train_df.groupby('YrSold')['SalePrice'].mean()
        plt.figure(figsize=(10, 6))
        yearly_sales.plot(kind='line', marker='o', color='red')
        plt.xlabel('Year')
        plt.ylabel('Average SalePrice')
        plt.title('Yearly Sales Trends')
        plt.grid(True)
        plt.savefig('visualizations/yearly_sales_trends.png')
        plt.close()

        # Scatter plot: Garage Area vs SalePrice
        plt.figure(figsize=(10, 6))
        plt.scatter(train_df['GarageArea'], train_df['SalePrice'], alpha=0.5, c='blue')
        plt.xlabel('Garage Area (m²)')
        plt.ylabel('SalePrice')
        plt.title('Price vs Garage Area')
        plt.grid(True)
        plt.savefig('visualizations/price_vs_garage_area.png')
        plt.close()

        print("✅All graphs were generated and saved in the 'visualizations' folder.")
