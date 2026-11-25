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

## Requirements
- Python 3.9 or later
- VS Code (recommended)
- Internet access only for installing Python packages

## Run in VS Code

1. Download or clone this repository.
2. Open the project folder in VS Code.
3. Open the VS Code terminal (`Terminal` → `New Terminal`).
4. Create a virtual environment:

   **Windows**
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

6. Run the project:
   ```bash
   python main.py
   ```

The script prints the evaluation metrics and creates the charts and CSV forecast files in the `outputs/` folder.

## Methodology

### 1. Data cleaning
- Parse dates and numeric sales values.
- Remove duplicate dates.
- Sort observations chronologically.
- Fill missing sales values using time interpolation.

### 2. Feature engineering
The models use:
- Calendar features: day of week, month, and day of year.
- Lag features: sales from 1, 7, and 14 days earlier.
- Rolling averages: previous 7-day and 14-day mean sales.
- A time index to represent the overall trend.

All lag and rolling features use only observations available before the date being predicted.

### 3. Model evaluation
The data is split chronologically: the earliest 80% of usable observations are used for training and the latest 20% for testing. The test period is not randomly shuffled.

Models:
- Seasonal Naive (7-day): uses sales from the same weekday one week earlier.
- Linear Regression.
- Random Forest Regressor.

Metrics:
- **MAE:** average absolute prediction error; lower is better.
- **RMSE:** gives larger errors more weight; lower is better.
- **R²:** indicates the proportion of target variation explained by the model; higher is generally better.

The model with the lowest holdout MAE is selected for the 30-day recursive forecast.

### 4. Forecasting
The selected regression model predicts one day ahead. Each predicted value is added to the history and used to construct the next day's lag and rolling features. This process repeats for 30 days.

## Outputs
After running `main.py`, see:
- `sales_trend.png` — historical daily sales.
- `actual_vs_predicted.png` — holdout actual values and model predictions.
- `future_forecast.png` — recent history and the next 30 days.
- `model_comparison.png` — MAE and RMSE comparison.
- `model_metrics.csv` — numerical evaluation results.
- `future_30_day_forecast.csv` — forecast values by date.

## Important limitations
- The dataset is synthetic and designed to demonstrate a workflow.
- Synthetic patterns may be easier to predict than real business sales.
- Recursive forecasts can accumulate errors over a longer horizon.
- The model does not include promotions, holidays, stock-outs, pricing, or external economic factors.
- Model performance on this dataset should not be interpreted as expected performance on real company data.

## Conclusion
This project demonstrates an end-to-end predictive analytics workflow: data preparation, time-series feature engineering, regression modelling, chronological evaluation, model comparison, and future forecasting. The workflow can be adapted to genuine business sales data after validating its data quality and forecasting requirements.

## Author
Add your name, internship details, and submission date here before submitting.

## License
This project is provided for educational and internship purposes.
