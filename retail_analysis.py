#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retail transaction analysis and revenue forecasting.

Cleans transaction data with PySpark, then uses pandas to find top products and
customers, decompose daily revenue, compare ARIMA and LSTM forecasts, and mine
product association rules (market basket analysis).

Usage:
    python retail_analysis.py

Expects transactionrecord.csv (columns: TransactionNo, Date, ProductName, Price,
Quantity, CustomerNo) in the working directory.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import itertools


# ========== Utility Functions ==========

def init_spark(app_name: str = "Data Wrangling with PySpark") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_data(spark, file_path: str):
    return spark.read.csv(file_path, header=True, inferSchema=True)


def clean_and_process(df):
    df = df.fillna({'CustomerNo': '-1'})
    df = df.withColumn('ProductName_process',
                        regexp_replace(col('ProductName'), '[^a-zA-Z]', ''))
    return df


def calculate_revenue(df):
    df = df.withColumn('revenue', (col('Price') * col('Quantity')).cast('float'))
    return df


def to_pandas(df):
    pdf = df.toPandas()
    pdf['transaction_date'] = pd.to_datetime(pdf['Date'], format='%m/%d/%Y')
    return pdf


def daily_revenue(df) -> pd.Series:
    """Total revenue per calendar day, with days that have no sales filled with the mean."""
    daily = df.groupby('transaction_date')['revenue'].sum().asfreq('D')
    return daily.fillna(daily.mean())


def top_n(df, by: str, n: int = 10) -> pd.Series:
    return df.groupby(by)['revenue'].sum().sort_values(ascending=False).head(n)


def plot_revenue_over_time(daily: pd.Series):
    plt.figure(figsize=(10, 6))
    plt.plot(daily.index, daily.values)
    plt.title('Revenue by Transaction Date')
    plt.xlabel('Transaction Date')
    plt.ylabel('Revenue')
    plt.grid(True)
    plt.show()


def decompose_time_series(daily: pd.Series):
    result = seasonal_decompose(daily, model='additive', period=7)
    result.plot()
    plt.show()


def arima_grid_search(daily: pd.Series, train_end: str):
    train = daily[daily.index < train_end]
    test = daily[daily.index >= train_end]
    pdq = list(itertools.product(range(0, 3), repeat=3))
    results = []

    for param in pdq:
        try:
            fit = ARIMA(train, order=param).fit()
            pred = fit.forecast(steps=len(test))
            pred.index = test.index
            results.append((param, mean_absolute_error(test, pred)))
        except Exception:
            continue

    if not results:
        print('No ARIMA configuration converged.')
        return
    best_param, best_mae = min(results, key=lambda x: x[1])
    print(f'Best ARIMA params: {best_param} with MAE: {best_mae:,.2f}')


def lstm_forecast(daily: pd.Series, look_back: int = 30):
    values = daily.values.reshape(-1, 1)
    train_size = int(len(values) * 0.8)

    # Fit the scaler on the training period only so the test period stays unseen
    scaler = MinMaxScaler().fit(values[:train_size])
    scaled = scaler.transform(values)
    train, test = scaled[:train_size], scaled[train_size - look_back:]

    def create_dataset(data):
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i+look_back])
            Y.append(data[i+look_back])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train)
    X_test, y_test = create_dataset(test)

    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Report error in revenue units, comparable with the ARIMA MAE
    y_pred = scaler.inverse_transform(model.predict(X_test))
    y_true = scaler.inverse_transform(y_test)
    print(f'LSTM MAE: {mean_absolute_error(y_true, y_pred):,.2f}')


def basket_analysis(df, top_products: int = 50, min_support: float = 0.02):
    """Association rules between products bought in the same transaction."""
    sales = df[df['Quantity'] > 0]
    popular = sales['ProductName'].value_counts().head(top_products).index
    basket = (sales[sales['ProductName'].isin(popular)]
              .groupby(['TransactionNo', 'ProductName'])['Quantity'].sum()
              .unstack(fill_value=0) > 0)

    itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        print('No frequent itemsets at this support level.')
        return
    rules = association_rules(itemsets, metric='lift', min_threshold=1.0)
    print(rules.sort_values('lift', ascending=False)
          [['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))


# ========== Main Execution ==========

if __name__ == "__main__":
    spark = init_spark()

    df = load_data(spark, 'transactionrecord.csv')
    df = clean_and_process(df)
    df = calculate_revenue(df)

    df_pandas = to_pandas(df)
    print(df_pandas[['CustomerNo', 'ProductName', 'ProductName_process', 'transaction_date', 'revenue']].head())

    print('\nTop products by revenue:\n', top_n(df_pandas, 'ProductName'))
    print('\nTop customers by revenue:\n', top_n(df_pandas, 'CustomerNo'))

    daily = daily_revenue(df_pandas)
    plot_revenue_over_time(daily)
    decompose_time_series(daily)

    arima_grid_search(daily, '2019-11-01')
    lstm_forecast(daily)

    basket_analysis(df_pandas)

    print("Analysis complete.")
