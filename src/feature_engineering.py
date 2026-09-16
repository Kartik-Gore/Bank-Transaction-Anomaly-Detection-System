"""
src/feature_engineering.py
Extracts customer behavioral profiles, velocity metrics, statistical z-scores,
and temporal indicators to prepare transaction datasets for machine learning.
"""

import pandas as pd
import numpy as np

def extract_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Transforms raw banking transactions into an enriched feature matrix.
    Returns:
        - enriched_df: DataFrame with original data + extracted features
        - feature_cols: List of column names used for ML model training
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
    df.sort_values(by=["customer_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1. Temporal Features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if 0 <= h <= 5 else 0)

    # 2. Customer Spending Profile (Baseline)
    cust_stats = df.groupby("customer_id")["amount"].agg(
        cust_mean_amount="mean",
        cust_std_amount="std",
        cust_median_amount="median",
        cust_max_amount="max",
        cust_txn_count="count"
    ).reset_index()

    cust_stats["cust_std_amount"] = cust_stats["cust_std_amount"].fillna(0.0)
    df = pd.merge(df, cust_stats, on="customer_id", how="left")

    # Amount Deviation Metrics
    df["amount_to_mean_ratio"] = df["amount"] / np.maximum(df["cust_mean_amount"], 1.0)
    std_safe = np.where(df["cust_std_amount"] < 1e-2, 1.0, df["cust_std_amount"])
    df["amount_zscore"] = (df["amount"] - df["cust_mean_amount"]) / std_safe
    df["amount_zscore"] = df["amount_zscore"].clip(lower=-5, upper=25)
    df["amount_diff_from_median"] = np.abs(df["amount"] - df["cust_median_amount"])

    # 3. Customer Velocity & Frequency Tracking
    df["prev_timestamp"] = df.groupby("customer_id")["timestamp"].shift(1)
    df["time_delta_mins"] = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 60.0
    df["time_delta_mins"] = df["time_delta_mins"].fillna(10080.0)
    df["rapid_succession_flag"] = (df["time_delta_mins"] < 15.0).astype(int)

    # Rolling window metrics
    df_indexed = df.set_index("timestamp")
    txns_1h = df_indexed.groupby("customer_id")["amount"].rolling("1h").count().reset_index()
    txns_1h.rename(columns={"amount": "txns_last_1h"}, inplace=True)
    
    rolling_24h = df_indexed.groupby("customer_id")["amount"].rolling("24h").agg(["count", "sum"]).reset_index()
    rolling_24h.rename(columns={"count": "txns_last_24h", "sum": "spend_last_24h"}, inplace=True)

    df["txns_last_1h"] = txns_1h["txns_last_1h"].values
    df["txns_last_24h"] = rolling_24h["txns_last_24h"].values
    df["spend_last_24h"] = rolling_24h["spend_last_24h"].values

    # 4. Merchant Category & Channel Risk Indicators
    high_risk_categories = {"Crypto & FX", "Gambling", "Luxury"}
    df["is_high_risk_category"] = df["merchant_category"].apply(
        lambda cat: 1 if cat in high_risk_categories else 0
    )
    
    df["is_transfer_type"] = df["transaction_type"].apply(
        lambda t: 1 if "Transfer" in str(t) or "Withdrawal" in str(t) else 0
    )

    df["log_amount"] = np.log1p(df["amount"])

    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [
        "amount",
        "log_amount",
        "amount_to_mean_ratio",
        "amount_zscore",
        "time_delta_mins",
        "rapid_succession_flag",
        "txns_last_1h",
        "txns_last_24h",
        "spend_last_24h",
        "hour_of_day",
        "is_weekend",
        "is_night",
        "is_high_risk_category",
        "is_transfer_type"
    ]

    return df, feature_cols
