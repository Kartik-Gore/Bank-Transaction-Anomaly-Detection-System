"""
src/data_generator.py
Synthesizes realistic banking transaction datasets with configurable volume
and diverse anomaly patterns (amount spikes, rapid velocity bursts, high-risk merchant switches,
unusual hours, and geographic jumps).
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def generate_bank_transactions(
    n_transactions: int = 5000,
    n_customers: int = 150,
    anomaly_rate: float = 0.04,
    random_state: int = 42,
    save_path: str = None
) -> pd.DataFrame:
    """
    Generates a realistic transaction dataset with genuine customer profiles
    and injected behavioral anomalies.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    # 1. Generate Customer Profiles (baseline behavior)
    customer_ids = [f"CUST_{1000 + i}" for i in range(n_customers)]
    customer_profiles = {}
    
    cities = [
        "New York, US", "Los Angeles, US", "Chicago, US", "Houston, US", 
        "Phoenix, US", "Philadelphia, US", "San Antonio, US", "San Diego, US",
        "Dallas, US", "San Jose, US", "London, UK", "Toronto, CA"
    ]
    
    merchant_categories = [
        "Groceries", "Dining", "Electronics", "Retail", "Utilities", 
        "Entertainment", "Healthcare", "Travel", "Crypto & FX", "Gambling"
    ]
    
    channels = ["Mobile App", "Web Banking", "POS Terminal", "ATM", "Branch"]
    transaction_types = ["POS Purchase", "Online Payment", "Bank Transfer", "ATM Withdrawal"]

    for cid in customer_ids:
        avg_spend = np.random.uniform(25.0, 180.0)
        spend_std = avg_spend * np.random.uniform(0.2, 0.5)
        home_city = random.choice(cities[:10]) # domestic home city
        favorite_categories = random.sample(merchant_categories[:8], k=4)
        preferred_channel = random.choice(channels[:3])
        
        customer_profiles[cid] = {
            "avg_spend": avg_spend,
            "spend_std": spend_std,
            "home_city": home_city,
            "favorite_categories": favorite_categories,
            "preferred_channel": preferred_channel
        }

    # 2. Generate Base Transactions across a 60-day window
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    total_seconds = int((end_date - start_date).total_seconds())

    records = []
    
    for i in range(n_transactions):
        txn_id = f"TXN_{100000 + i}"
        cid = random.choice(customer_ids)
        profile = customer_profiles[cid]
        
        rand_sec = np.random.randint(0, total_seconds)
        dt = start_date + timedelta(seconds=rand_sec)
        
        # Daytime bias for normal transactions
        if random.random() < 0.85:
            hour = np.random.randint(8, 23)
            dt = dt.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        
        amount = max(2.5, np.random.normal(profile["avg_spend"], profile["spend_std"]))
        amount = round(amount, 2)
        
        category = random.choice(profile["favorite_categories"]) if random.random() < 0.80 else random.choice(merchant_categories[:8])
        channel = profile["preferred_channel"] if random.random() < 0.75 else random.choice(channels)
        txn_type = random.choice(transaction_types)
        location = profile["home_city"] if random.random() < 0.90 else random.choice(cities)
        
        records.append({
            "transaction_id": txn_id,
            "customer_id": cid,
            "timestamp": dt,
            "amount": amount,
            "transaction_type": txn_type,
            "merchant_category": category,
            "channel": channel,
            "location": location,
            "is_anomaly": 0,
            "anomaly_type": "Normal"
        })

    df = pd.DataFrame(records)
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3. Inject Realistic Anomalies based on anomaly_rate
    num_anomalies = int(n_transactions * anomaly_rate)
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    
    anomaly_types = [
        "Amount Spike",
        "High Velocity Burst",
        "High Risk Merchant",
        "Unusual Night Transfer",
        "Geographical Jump"
    ]

    for idx in anomaly_indices:
        atype = random.choice(anomaly_types)
        cid = df.at[idx, "customer_id"]
        profile = customer_profiles[cid]
        
        if atype == "Amount Spike":
            multiplier = np.random.uniform(10.0, 35.0)
            df.at[idx, "amount"] = round(profile["avg_spend"] * multiplier, 2)
            df.at[idx, "transaction_type"] = "Bank Transfer"
            df.at[idx, "is_anomaly"] = 1
            df.at[idx, "anomaly_type"] = "Amount Spike"

        elif atype == "High Velocity Burst":
            df.at[idx, "is_anomaly"] = 1
            df.at[idx, "anomaly_type"] = "High Velocity Burst"
            base_time = df.at[idx, "timestamp"]
            for offset in range(1, 4):
                if idx + offset < len(df):
                    df.at[idx + offset, "customer_id"] = cid
                    df.at[idx + offset, "timestamp"] = base_time + timedelta(minutes=random.randint(1, 4) * offset)
                    df.at[idx + offset, "amount"] = round(np.random.uniform(400, 1200), 2)
                    df.at[idx + offset, "transaction_type"] = "Online Payment"
                    df.at[idx + offset, "is_anomaly"] = 1
                    df.at[idx + offset, "anomaly_type"] = "High Velocity Burst"

        elif atype == "High Risk Merchant":
            df.at[idx, "merchant_category"] = random.choice(["Crypto & FX", "Gambling"])
            df.at[idx, "amount"] = round(profile["avg_spend"] * np.random.uniform(5.0, 15.0), 2)
            df.at[idx, "channel"] = "Web Banking"
            df.at[idx, "is_anomaly"] = 1
            df.at[idx, "anomaly_type"] = "High Risk Merchant"

        elif atype == "Unusual Night Transfer":
            ts = df.at[idx, "timestamp"]
            df.at[idx, "timestamp"] = ts.replace(hour=random.choice([2, 3, 4]), minute=random.randint(0, 59))
            df.at[idx, "transaction_type"] = "Bank Transfer"
            df.at[idx, "amount"] = round(profile["avg_spend"] * np.random.uniform(6.0, 18.0), 2)
            df.at[idx, "is_anomaly"] = 1
            df.at[idx, "anomaly_type"] = "Unusual Night Transfer"

        elif atype == "Geographical Jump":
            df.at[idx, "location"] = random.choice(["London, UK", "Toronto, CA", "Dubai, UAE", "Tokyo, JP"])
            df.at[idx, "channel"] = "ATM"
            df.at[idx, "amount"] = round(profile["avg_spend"] * np.random.uniform(4.0, 12.0), 2)
            df.at[idx, "is_anomaly"] = 1
            df.at[idx, "anomaly_type"] = "Geographical Jump"

    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Determine save path if requested
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        df.to_csv(save_path, index=False)
    else:
        # Default destination: data/transactions.csv
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_dest = os.path.join(base_dir, "data", "transactions.csv")
        os.makedirs(os.path.dirname(default_dest), exist_ok=True)
        df.to_csv(default_dest, index=False)

    return df

if __name__ == "__main__":
    print("Generating sample bank transactions...")
    data = generate_bank_transactions(n_transactions=3000, anomaly_rate=0.04)
    print(f"Generated {len(data)} transactions with {data['is_anomaly'].sum()} injected anomalies.")
