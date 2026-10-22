# Retail Transaction Analysis & Forecasting

An end-to-end retail analytics pipeline: clean raw transactions at scale with **PySpark**, find the products and customers that drive revenue, forecast daily sales with **ARIMA** and an **LSTM** network, and uncover which products are bought together.

## What it answers

| Business question | Method |
| --- | --- |
| Which products and customers bring in the most revenue? | Revenue = price × quantity, ranked top 10 |
| How does revenue move over time? Is there a weekly pattern? | Daily revenue series + additive seasonal decomposition |
| What will revenue look like next? | ARIMA (grid search over p, d, q) vs. LSTM, compared on MAE |
| Which products are bought together? | Market basket analysis — Apriori + association rules ranked by lift |

## Pipeline

1. **Wrangle with PySpark** — load the CSV, fill missing customer IDs, normalise product names, compute line revenue.
2. **Aggregate** — convert to pandas and build a continuous daily revenue series.
3. **Explore** — top products, top customers, revenue trend, weekly seasonality.
4. **Forecast** — tune ARIMA on data before November 2019 and test on the rest; train a 30-day look-back LSTM on the first 80% of days. Both report MAE in revenue units.
5. **Basket analysis** — one-hot encode the 50 most popular products per transaction and mine association rules.

## Getting started

```bash
git clone https://github.com/kr-xna/retail-transaction-analysis.git
cd retail-transaction-analysis
pip install -r requirements.txt
python retail_analysis.py
```

PySpark needs Java 8 or later installed. Place `transactionrecord.csv` in the project folder first, with columns `TransactionNo`, `Date` (MM/DD/YYYY), `ProductName`, `Price`, `Quantity` and `CustomerNo`. The dataset is not included in this repository.

## Tech stack

Python · PySpark · pandas · statsmodels · TensorFlow / Keras · mlxtend · scikit-learn · Matplotlib
