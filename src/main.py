"""
src/main.py
CLI entrypoint to run end-to-end Bank Transaction Anomaly Detection,
generate summary metrics, export an audit report CSV to data/, and produce visual plots to outputs/.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_generator import generate_bank_transactions
from feature_engineering import extract_features
from anomaly_detector import BankAnomalyDetector

def run_pipeline(
    csv_path: str = None,
    algorithm: str = "Isolation Forest",
    contamination: float = 0.05
):
    print("=" * 65)
    print("       BANK TRANSACTION ANOMALY DETECTION ENGINE        ")
    print("=" * 65)

    data_dir = os.path.join(project_root, "data")
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    if csv_path is None:
        csv_path = os.path.join(data_dir, "transactions.csv")

    # 1. Load or Generate Dataset
    if os.path.exists(csv_path):
        print(f"[1/5] Ingesting transaction data from '{csv_path}'...")
        df_raw = pd.read_csv(csv_path)
    else:
        print(f"[1/5] '{csv_path}' not found. Generating synthetic bank transactions...")
        df_raw = generate_bank_transactions(
            n_transactions=3500,
            anomaly_rate=0.04,
            save_path=csv_path
        )

    print(f"      Total records loaded: {len(df_raw):,} transactions")
    print(f"      Unique customers:     {df_raw['customer_id'].nunique():,}")
    print(f"      Total transaction volume: ${df_raw['amount'].sum():,.2f}")

    # 2. Feature Engineering
    print("\n[2/5] Running customer behavioral profiling & feature engineering...")
    df_feat, feature_cols = extract_features(df_raw)
    print(f"      Generated {len(feature_cols)} ML features:")
    print(f"      -> {', '.join(feature_cols[:6])}...")

    # 3. Anomaly Detection
    print(f"\n[3/5] Training {algorithm} (contamination={contamination:.1%})...")
    detector = BankAnomalyDetector(
        algorithm=algorithm,
        contamination=contamination,
        random_state=42
    )
    df_results = detector.fit_predict(df_feat, feature_cols)

    flagged = df_results[df_results["is_detected_anomaly"] == 1]
    at_risk_amount = flagged["amount"].sum()
    print(f"\n[4/5] Anomaly Detection Summary:")
    print(f"      Flagged Anomalies:      {len(flagged):,} ({len(flagged)/len(df_results):.2%})")
    print(f"      Total At-Risk Capital:  ${at_risk_amount:,.2f}")
    print(f"      Avg Flagged Amount:     ${flagged['amount'].mean():,.2f} (vs normal avg ${df_results[df_results['is_detected_anomaly']==0]['amount'].mean():,.2f})")

    # Evaluation if ground truth exists
    metrics = detector.evaluate_performance(df_results)
    if metrics:
        print("\n      Validation vs Injected Ground Truth:")
        print(f"      - Precision: {metrics['precision']:.2%}")
        print(f"      - Recall:    {metrics['recall']:.2%}")
        print(f"      - F1 Score:  {metrics['f1_score']:.2%}")

    # 4. Export CSV Audit Report
    report_cols = [
        "transaction_id", "customer_id", "timestamp", "amount",
        "transaction_type", "merchant_category", "channel", "location",
        "anomaly_score", "risk_level", "anomaly_reasons"
    ]
    report_df = flagged[report_cols].sort_values(by="anomaly_score", ascending=False)
    report_filename = os.path.join(data_dir, "anomaly_report.csv")
    report_df.to_csv(report_filename, index=False)
    print(f"\n[5/5] Exported audit report with {len(report_df)} flagged cases to '{report_filename}'")

    # 5. Generate Visual Summary (Matplotlib & Seaborn)
    chart_output = os.path.join(outputs_dir, "anomaly_analysis.png")
    generate_visual_plots(df_results, output_path=chart_output)
    print(f"      Generated visualization summary chart: '{chart_output}'")
    print("=" * 65)
    print("Pipeline completed successfully! Launch web UI with: streamlit run dashboard/app.py\n")

def generate_visual_plots(df: pd.DataFrame, output_path: str):
    """Generates a 4-panel publication-ready visualization dashboard using Matplotlib/Seaborn."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Bank Transaction Anomaly Analysis Dashboard", fontsize=18, fontweight="bold", y=0.98)

    palette = {0: "#2b5c8f", 1: "#d9383a"}

    # Panel 1: Anomaly Score Distribution
    sns.histplot(
        data=df,
        x="anomaly_score",
        hue="is_detected_anomaly",
        palette=palette,
        bins=40,
        kde=True,
        ax=axes[0, 0],
        alpha=0.6
    )
    axes[0, 0].set_title("1. Distribution of Composite Anomaly Scores", fontsize=13, fontweight="semibold")
    axes[0, 0].set_xlabel("Anomaly Score (0 - 100)")
    axes[0, 0].set_ylabel("Transaction Count")
    axes[0, 0].legend(["Anomaly", "Normal"], title="Status")

    # Panel 2: Amount vs Time Delta (Velocity)
    sample_df = df.sample(min(1500, len(df)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x="time_delta_mins",
        y="amount",
        hue="is_detected_anomaly",
        palette=palette,
        alpha=0.75,
        s=45,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title("2. Transaction Amount vs Minutes Since Prior Transaction", fontsize=13, fontweight="semibold")
    axes[0, 1].set_xlabel("Time Since Customer's Last Transaction (Minutes)")
    axes[0, 1].set_ylabel("Amount ($)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(["Normal", "Anomaly"], title="Status")

    # Panel 3: Risk Level Breakdown
    risk_counts = df["risk_level"].value_counts().reindex(["Low Risk", "Moderate Risk", "Elevated Risk", "Critical Risk"]).fillna(0)
    bar_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    axes[1, 0].bar(risk_counts.index, risk_counts.values, color=bar_colors, edgecolor="black", linewidth=0.8)
    axes[1, 0].set_title("3. Transaction Count by Risk Classification", fontsize=13, fontweight="semibold")
    axes[1, 0].set_ylabel("Number of Transactions")
    for idx, val in enumerate(risk_counts.values):
        axes[1, 0].text(idx, val + (max(risk_counts.values)*0.02), f"{int(val):,}", ha="center", fontweight="bold")

    # Panel 4: Anomalies by Merchant Category
    cat_summary = df.groupby("merchant_category")["is_detected_anomaly"].agg(
        total="count",
        anomalies="sum"
    ).reset_index()
    cat_summary["anomaly_pct"] = (cat_summary["anomalies"] / cat_summary["total"]) * 100
    cat_summary.sort_values(by="anomaly_pct", ascending=True, inplace=True)

    axes[1, 1].barh(cat_summary["merchant_category"], cat_summary["anomaly_pct"], color="#e74c3c", alpha=0.85)
    axes[1, 1].set_title("4. Anomaly Rate by Merchant Category (%)", fontsize=13, fontweight="semibold")
    axes[1, 1].set_xlabel("Percentage Flagged as Anomaly (%)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    run_pipeline()
