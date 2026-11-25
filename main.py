"""Main entry point for the Predictive Analytics internship project."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_cleaning import load_and_clean_data
from model_training import evaluate_models, recursive_forecast
from visualization import save_visualizations

def main():
    data_path = ROOT / "data" / "historical_sales.csv"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("=" * 65)
    print("PREDICTIVE ANALYTICS USING HISTORICAL DATA")
    print("Application: Daily Sales Forecasting")
    print("=" * 65)

    df = load_and_clean_data(data_path)
    sales = df["Sales"]

    print(f"\nCleaned observations: {len(sales):,}")
    print(f"Date range: {sales.index.min().date()} to {sales.index.max().date()}")
    print(f"Missing values after cleaning: {int(sales.isna().sum())}")

    metrics, predictions, y_test, best_name, best_model, _ = evaluate_models(sales)
    print("\nMODEL EVALUATION (chronological holdout)")
    print(metrics.round(2).to_string())

    forecast = recursive_forecast(sales, best_model, horizon=30)
    forecast.to_csv(output_dir / "future_30_day_forecast.csv", index=False)
    metrics.to_csv(output_dir / "model_metrics.csv")

    save_visualizations(sales, predictions, y_test, forecast, metrics, output_dir)

    print(f"\nBest model by MAE: {best_name}")
    print(f"Forecast saved: {output_dir / 'future_30_day_forecast.csv'}")
    print(f"Charts saved in: {output_dir}")
    print("\nNote: The included dataset is synthetic and for educational use.")
    print("Forecasts are illustrative, not business or financial advice.")

if __name__ == "__main__":
    main()
