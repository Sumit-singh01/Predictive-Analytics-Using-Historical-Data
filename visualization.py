"""Save charts for the project report and GitHub repository."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def save_visualizations(series, predictions, y_test, forecast, metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values, linewidth=1, label="Historical sales")
    plt.title("Historical Daily Sales Trend")
    plt.xlabel("Date"); plt.ylabel("Sales")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "sales_trend.png", dpi=160); plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test.values, label="Actual", linewidth=1.5)
    for name, pred in predictions.items():
        plt.plot(pred.index, pred.values, label=name, alpha=0.8)
    plt.title("Actual vs Predicted Sales (Chronological Test Period)")
    plt.xlabel("Date"); plt.ylabel("Sales")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=160); plt.close()

    plt.figure(figsize=(12, 5))
    recent = series.iloc[-90:]
    plt.plot(recent.index, recent.values, label="Last 90 days", linewidth=1.5)
    plt.plot(forecast["Date"], forecast["Predicted_Sales"],
             label="Next 30-day forecast", linestyle="--", linewidth=2)
    plt.title("30-Day Sales Forecast")
    plt.xlabel("Date"); plt.ylabel("Sales")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "future_forecast.png", dpi=160); plt.close()

    metric_plot = metrics[["MAE", "RMSE"]]
    ax = metric_plot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Model Comparison: MAE and RMSE (lower is better)")
    ax.set_ylabel("Error"); ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.25); plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160); plt.close()
