# Big Data-Driven Seasonal & Regional Product Demand Prediction

## Project Overview

A machine learning-based product demand prediction system designed to help shopkeepers and retail business owners make better inventory planning decisions.

Users select a country and season to receive the top 10 products with the highest predicted demand.

## Technologies Used

- Python
- Apache PySpark
- Pandas and NumPy
- Scikit-learn
- Random Forest Regressor
- Flask
- HTML, CSS and JavaScript

## Dataset

Online Retail II dataset containing historical retail transaction records.

The dataset includes product descriptions, quantities, invoice dates, prices and countries.

**Note:** The dataset is excluded from this repository because of its large file size. Place `online_retail_II.csv` in the project root before running the application.

## Data Processing

1. Load data using PySpark.
2. Remove records with missing Description and Country.
3. Filter transactions with non-positive quantities.
4. Extract months from InvoiceDate.
5. Derive the Season feature.
6. Aggregate quantities by Country, Season and Description.
7. Encode categorical features.

## Seasonal Mapping

| Months | Season |
|---|---|
| March – June | Summer |
| July – September | Monsoon |
| October – February | Winter |

## Machine Learning

**Model:** RandomForestRegressor

**Input features:**
- Country
- Season
- Product

**Target:** Aggregated Quantity

Training and testing split: 80:20.

## Application Workflow

1. Select a country.
2. Select a season.
3. Click Predict Products.
4. View the top 10 recommended products.

## Installation

Clone the repository:

```bash
git clone https://github.com/karunjoe2404-coder/ProductDemand-.git
cd ProductDemand-
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place `online_retail_II.csv` in the project root.

Run the application:

```bash
python bdapy.py
```

Open `http://127.0.0.1:5000/` in your browser.

## Future Enhancements

- Model comparison and hyperparameter tuning.
- Improved time-aware demand forecasting.
- Interactive analytics dashboards.
- Country-specific seasonal definitions.
- Cloud deployment.

## Author

**Karun Joe Sarvesh S**

M.Sc. Data Science  
Loyola College, Chennai