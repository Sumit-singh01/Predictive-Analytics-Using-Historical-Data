# Predictive Analytics Using Historical Data
### Daily Sales Forecasting with Python and Machine Learning

## Project overview
This project demonstrates how historical daily sales data can be cleaned, explored, modelled, evaluated, and used to generate a 30-day forecast. It compares a seasonal-naive baseline with Linear Regression and Random Forest models.

**Dataset notice:** `data/historical_sales.csv` is synthetic data generated for this educational project. It does not represent a real company or actual customer transactions.

## Objectives
- Clean and preprocess historical time-series data.
- Explore daily sales trends and patterns.
- Build regression models using historical and calendar features.
- Evaluate models using MAE, RMSE, and R-squared.
- Compare model performance against a simple 7-day seasonal-naive baseline.
- Forecast the next 30 days and visualize the results.

## Project structure
```text
Predictive-Analytics-Using-Historical-Data/
├── data/
│   └── historical_sales.csv
├── outputs/
│   ├── sales_trend.png
│   ├── actual_vs_predicted.png
│   ├── future_forecast.png
│   ├── model_comparison.png
│   ├── model_metrics.csv
│   └── future_30_day_forecast.csv
├── src/
│   ├── data_cleaning.py
│   ├── model_training.py
│   └── visualization.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```


