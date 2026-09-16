"""
dashboard/app.py
Streamlit Web Application for Bank Transaction Anomaly Detection.
Provides interactive visual dashboards, customer profiling, ML model tuning,
transaction autopsy drilldowns, and downloadable audit reports.
"""

import os
import sys
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set up paths so dashboard can import from src/ and locate data/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_generator import generate_bank_transactions
from feature_engineering import extract_features
from anomaly_detector import BankAnomalyDetector

# ---------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="BankGuard AI | Transaction Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern financial UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-elevated {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-moderate {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .reason-box {
        background-color: #FFF1F2;
        border-left: 4px solid #E11D48;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls & Data Management
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
st.sidebar.title("BankGuard AI")
st.sidebar.caption("Intelligent Anti-Fraud & Anomaly Sentinel")

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Source")
data_option = st.sidebar.radio(
    "Select Transaction Data:",
    ["Use Existing Data (data/transactions.csv)", "Generate New Synthetic Data", "Upload CSV"]
)

default_csv_path = os.path.join(data_dir, "transactions.csv")
raw_df = None

if data_option == "Use Existing Data (data/transactions.csv)":
    if os.path.exists(default_csv_path):
        raw_df = pd.read_csv(default_csv_path)
        st.sidebar.success(f"Loaded {len(raw_df):,} transactions from data/transactions.csv")
    else:
        st.sidebar.warning("data/transactions.csv not found. Generating default data...")
        raw_df = generate_bank_transactions(n_transactions=3000, anomaly_rate=0.04, save_path=default_csv_path)

elif data_option == "Generate New Synthetic Data":
    n_txns = st.sidebar.slider("Number of Transactions", 500, 10000, 3000, step=500)
    n_cust = st.sidebar.slider("Number of Customers", 50, 500, 150, step=25)
    rate = st.sidebar.slider("Anomaly Injection Rate", 0.01, 0.15, 0.04, step=0.01)
    if st.sidebar.button("🔄 Generate Fresh Dataset", use_container_width=True):
        raw_df = generate_bank_transactions(
            n_transactions=n_txns, n_customers=n_cust, anomaly_rate=rate, save_path=default_csv_path
        )
        st.sidebar.success(f"Generated {len(raw_df):,} fresh transactions!")
    else:
        if os.path.exists(default_csv_path):
            raw_df = pd.read_csv(default_csv_path)
        else:
            raw_df = generate_bank_transactions(
                n_transactions=n_txns, n_customers=n_cust, anomaly_rate=rate, save_path=default_csv_path
            )

elif data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload bank transaction CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Uploaded {len(raw_df):,} records!")
        except Exception as err:
            st.sidebar.error(f"Error loading file: {err}")
    else:
        st.sidebar.info("Awaiting file upload. Using default sample data...")
        if os.path.exists(default_csv_path):
            raw_df = pd.read_csv(default_csv_path)
        else:
            raw_df = generate_bank_transactions(n_transactions=2000, anomaly_rate=0.04, save_path=default_csv_path)

# ---------------------------------------------------------
# Sidebar Model Hyperparameters
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Configuration")
algo_choice = st.sidebar.selectbox(
    "Algorithm:",
    ["Isolation Forest", "Local Outlier Factor"]
)
contamination_rate = st.sidebar.slider(
    "Expected Contamination (%):",
    min_value=1.0,
    max_value=15.0,
    value=5.0,
    step=0.5
) / 100.0

custom_threshold = st.sidebar.slider(
    "Min Score Threshold for Alert (0-100):",
    min_value=30,
    max_value=90,
    value=55,
    step=5
)

# ---------------------------------------------------------
# Feature Extraction & Model Execution (Cached for speed)
# ---------------------------------------------------------
@st.cache_data(show_spinner="Extracting customer behavioral profiles & engineering features...")
def cached_feature_extraction(df):
    return extract_features(df)

@st.cache_data(show_spinner="Running ML anomaly detection & calibration...")
def cached_model_prediction(df_feat, features, algo, contam):
    det = BankAnomalyDetector(algorithm=algo, contamination=contam)
    res = det.fit_predict(df_feat, features)
    metrics = det.evaluate_performance(res)
    return res, metrics

# Pipeline execution
df_features, feature_names = cached_feature_extraction(raw_df)
df_results, ground_truth_metrics = cached_model_prediction(
    df_features, feature_names, algo_choice, contamination_rate
)

# Apply dynamic threshold from slider
df_results["is_detected_anomaly"] = (df_results["anomaly_score"] >= custom_threshold).astype(int)

# ---------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------
st.markdown("<div class='main-header'>🛡️ Bank Transaction Anomaly Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Real-time behavioral profiling, velocity monitoring, and machine-learning fraud intelligence.</div>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_overview, tab_explorer, tab_customer, tab_model, tab_report = st.tabs([
    "📊 Executive Dashboard",
    "🔍 Transaction Explorer",
    "👤 Customer Profiling",
    "🧠 Model Intelligence",
    "📋 Audit Report"
])

# ---------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD
# ---------------------------------------------------------
with tab_overview:
    total_txns = len(df_results)
    total_volume = df_results["amount"].sum()
    flagged_txns = df_results[df_results["is_detected_anomaly"] == 1]
    n_anomalies = len(flagged_txns)
    anomaly_pct = (n_anomalies / total_txns) * 100
    at_risk_capital = flagged_txns["amount"].sum()
    avg_anomaly_amt = flagged_txns["amount"].mean() if n_anomalies > 0 else 0
    avg_normal_amt = df_results[df_results["is_detected_anomaly"] == 0]["amount"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value">{total_txns:,}</div>
            <div class="metric-sub" style="color:#0284c7;">Across {df_results['customer_id'].nunique():,} accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Volume Monitored</div>
            <div class="metric-value">${total_volume:,.2f}</div>
            <div class="metric-sub" style="color:#64748b;">Avg txn: ${total_volume/total_txns:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Flagged Anomalies</div>
            <div class="metric-value" style="color:#DC2626;">{n_anomalies:,}</div>
            <div class="metric-sub" style="color:#DC2626;"><strong>{anomaly_pct:.2f}%</strong> of total flow</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">At-Risk Capital</div>
            <div class="metric-value" style="color:#B91C1C;">${at_risk_capital:,.2f}</div>
            <div class="metric-sub" style="color:#B91C1C;">Avg anomaly: ${avg_anomaly_amt:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns([2.2, 1.2])

    with c_left:
        st.subheader("📈 Transaction Stream & Anomaly Outliers")
        plot_sample = df_results.sort_values(by="timestamp").copy()
        if len(plot_sample) > 2000:
            plot_sample = plot_sample.sample(2000, random_state=42).sort_values(by="timestamp")

        fig_timeline = px.scatter(
            plot_sample,
            x="timestamp",
            y="amount",
            color=plot_sample["is_detected_anomaly"].map({0: "Normal", 1: "Anomaly Flagged"}),
            color_discrete_map={"Normal": "#3B82F6", "Anomaly Flagged": "#EF4444"},
            size=plot_sample["anomaly_score"],
            hover_data=["transaction_id", "customer_id", "amount", "merchant_category", "anomaly_score"],
            title="Transaction Amount Timeline (Dot Size = Anomaly Score)",
            labels={"timestamp": "Transaction Time", "amount": "Amount ($)", "color": "Status"}
        )
        fig_timeline.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_timeline, use_container_width=True)

    with c_right:
        st.subheader("🎯 Risk Tier Distribution")
        risk_counts = df_results["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Tier", "Count"]
        color_discrete = {
            "Low Risk": "#22C55E",
            "Moderate Risk": "#EAB308",
            "Elevated Risk": "#F97316",
            "Critical Risk": "#EF4444"
        }
        fig_pie = px.pie(
            risk_counts,
            names="Risk Tier",
            values="Count",
            color="Risk Tier",
            color_discrete_map=color_discrete,
            hole=0.45,
            title="Transactions by Risk Classification"
        )
        fig_pie.update_layout(height=420)
        st.plotly_chart(fig_pie, use_container_width=True)

    c_h1, c_h2 = st.columns(2)
    with c_h1:
        st.subheader("📊 Anomaly Score Histogram")
        fig_score = px.histogram(
            df_results,
            x="anomaly_score",
            color=df_results["is_detected_anomaly"].map({0: "Normal", 1: "Anomaly"}),
            color_discrete_map={"Normal": "#38BDF8", "Anomaly": "#E11D48"},
            nbins=40,
            marginal="box",
            title="Calibrated Risk Score (0 = Completely Normal, 100 = Severe Anomaly)"
        )
        fig_score.add_vline(x=custom_threshold, line_dash="dash", line_color="red", annotation_text=f"Alert Threshold ({custom_threshold})")
        fig_score.update_layout(height=360)
        st.plotly_chart(fig_score, use_container_width=True)

    with c_h2:
        st.subheader("🏪 Anomalies by Merchant Category")
        cat_stats = df_results.groupby("merchant_category").agg(
            total_txns=("transaction_id", "count"),
            anomalies=("is_detected_anomaly", "sum")
        ).reset_index()
        cat_stats["anomaly_rate"] = (cat_stats["anomalies"] / cat_stats["total_txns"]) * 100
        cat_stats.sort_values(by="anomaly_rate", ascending=True, inplace=True)
        fig_cat = px.bar(
            cat_stats,
            y="merchant_category",
            x="anomaly_rate",
            orientation="h",
            color="anomaly_rate",
            color_continuous_scale="Reds",
            title="Anomaly Vulnerability Rate by Category (%)",
            labels={"anomaly_rate": "Anomaly Rate (%)", "merchant_category": "Merchant Category"}
        )
        fig_cat.update_layout(height=360)
        st.plotly_chart(fig_cat, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: TRANSACTION EXPLORER & AUTOPSY DRILLDOWN
# ---------------------------------------------------------
with tab_explorer:
    st.subheader("🔍 Transaction Investigation & Live Inspection")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level:",
            options=["All", "Critical Risk", "Elevated Risk", "Moderate Risk", "Low Risk"],
            default=["All"]
        )
    with f_col2:
        cat_filter = st.multiselect(
            "Merchant Category:",
            options=["All"] + sorted(list(df_results["merchant_category"].unique())),
            default=["All"]
        )
    with f_col3:
        channel_filter = st.multiselect(
            "Channel:",
            options=["All"] + sorted(list(df_results["channel"].unique())),
            default=["All"]
        )
    with f_col4:
        search_query = st.text_input("🔎 Search (Txn ID or Customer ID):", "")

    filtered_df = df_results.copy()
    if "All" not in risk_filter and len(risk_filter) > 0:
        filtered_df = filtered_df[filtered_df["risk_level"].isin(risk_filter)]
    if "All" not in cat_filter and len(cat_filter) > 0:
        filtered_df = filtered_df[filtered_df["merchant_category"].isin(cat_filter)]
    if "All" not in channel_filter and len(channel_filter) > 0:
        filtered_df = filtered_df[filtered_df["channel"].isin(channel_filter)]
    if search_query:
        q = search_query.strip().upper()
        filtered_df = filtered_df[
            filtered_df["transaction_id"].str.contains(q, na=False) |
            filtered_df["customer_id"].str.contains(q, na=False)
        ]

    st.write(f"Showing **{len(filtered_df):,}** matching transactions (Sorted by Anomaly Risk Score):")

    display_cols = [
        "transaction_id", "customer_id", "timestamp", "amount",
        "anomaly_score", "risk_level", "transaction_type",
        "merchant_category", "channel", "location"
    ]
    st.dataframe(
        filtered_df[display_cols].sort_values(by="anomaly_score", ascending=False),
        use_container_width=True,
        height=320
    )

    st.markdown("---")
    st.subheader("🔬 Transaction Behavioral Autopsy & Explainability")

    top_candidates = filtered_df.sort_values(by="anomaly_score", ascending=False)["transaction_id"].tolist()
    if top_candidates:
        selected_txn_id = st.selectbox(
            "Select a Transaction to Inspect Detailed Reasoning:",
            options=top_candidates,
            index=0
        )

        txn_record = df_results[df_results["transaction_id"] == selected_txn_id].iloc[0]

        d_col1, d_col2 = st.columns([1.2, 1.8])
        with d_col1:
            st.markdown("#### Transaction Summary")
            st.write(f"**Transaction ID:** `{txn_record['transaction_id']}`")
            st.write(f"**Customer ID:** `{txn_record['customer_id']}`")
            st.write(f"**Timestamp:** {txn_record['timestamp']}")
            st.write(f"**Amount:** **${txn_record['amount']:,.2f}**")
            st.write(f"**Category:** {txn_record['merchant_category']}")
            st.write(f"**Channel:** {txn_record['channel']} ({txn_record['transaction_type']})")
            st.write(f"**Location:** {txn_record['location']}")

            risk_tier = txn_record['risk_level']
            badge_class = {
                "Critical Risk": "badge-critical",
                "Elevated Risk": "badge-elevated",
                "Moderate Risk": "badge-moderate",
                "Low Risk": "badge-low"
            }.get(risk_tier, "badge-low")

            st.markdown(f"**Risk Level:** <span class='{badge_class}'>{risk_tier}</span>", unsafe_allow_html=True)
            st.markdown(f"**Anomaly Score:** `{txn_record['anomaly_score']:.1f} / 100.0`")

        with d_col2:
            st.markdown("#### Automated Explainability Reasons")
            reasons_list = txn_record["anomaly_reasons"].split(" | ")
            for r in reasons_list:
                st.markdown(f"""
                <div class="reason-box">
                    🚨 <strong>{r}</strong>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Customer Comparison Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Customer Avg Spend", f"${txn_record['cust_mean_amount']:.2f}")
            m2.metric("Amount Ratio to Mean", f"{txn_record['amount_to_mean_ratio']:.1f}x")
            m3.metric("Z-Score Deviation", f"+{txn_record['amount_zscore']:.2f}σ")
    else:
        st.info("No transactions match current filters.")

# ---------------------------------------------------------
# TAB 3: CUSTOMER BEHAVIORAL PROFILING
# ---------------------------------------------------------
with tab_customer:
    st.subheader("👤 Customer Baseline Profiling & Activity Timeline")

    cust_list = sorted(list(df_results["customer_id"].unique()))
    selected_cust = st.selectbox("Select Customer to Profile:", cust_list, index=0)

    cust_df = df_results[df_results["customer_id"] == selected_cust].sort_values(by="timestamp")

    cp1, cp2, cp3, cp4 = st.columns(4)
    with cp1:
        st.metric("Total Transactions", len(cust_df))
    with cp2:
        st.metric("Total Spend", f"${cust_df['amount'].sum():,.2f}")
    with cp3:
        st.metric("Average Spend", f"${cust_df['amount'].mean():,.2f}")
    with cp4:
        cust_anomalies = cust_df['is_detected_anomaly'].sum()
        st.metric("Flagged Anomalies", f"{cust_anomalies}", delta=f"{cust_anomalies} flagged", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    fig_cust = go.Figure()
    mean_val = cust_df["cust_mean_amount"].iloc[0]
    fig_cust.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Customer Mean Baseline: ${mean_val:.2f}",
        annotation_position="bottom right"
    )

    normal_cust = cust_df[cust_df["is_detected_anomaly"] == 0]
    fig_cust.add_trace(go.Scatter(
        x=normal_cust["timestamp"],
        y=normal_cust["amount"],
        mode="markers+lines",
        name="Normal Transaction",
        marker=dict(size=8, color="#2563EB"),
        line=dict(color="#93C5FD", width=1)
    ))

    anom_cust = cust_df[cust_df["is_detected_anomaly"] == 1]
    if len(anom_cust) > 0:
        fig_cust.add_trace(go.Scatter(
            x=anom_cust["timestamp"],
            y=anom_cust["amount"],
            mode="markers",
            name="Anomaly Flagged",
            marker=dict(size=14, color="#DC2626", symbol="diamond", line=dict(width=2, color="#7F1D1D"))
        ))

    fig_cust.update_layout(
        title=f"Transaction History & Anomaly Detection for Customer: {selected_cust}",
        xaxis_title="Date & Time",
        yaxis_title="Amount ($)",
        height=400
    )
    st.plotly_chart(fig_cust, use_container_width=True)

    c_cat1, c_cat2 = st.columns(2)
    with c_cat1:
        st.markdown("#### Spending by Category")
        cust_cat = cust_df.groupby("merchant_category")["amount"].sum().reset_index()
        fig_ccat = px.bar(cust_cat, x="merchant_category", y="amount", color="amount", color_continuous_scale="Viridis")
        fig_ccat.update_layout(height=320)
        st.plotly_chart(fig_ccat, use_container_width=True)

    with c_cat2:
        st.markdown("#### Velocity & Succession Intervals")
        fig_vel = px.scatter(
            cust_df,
            x="time_delta_mins",
            y="amount",
            color="is_detected_anomaly",
            color_discrete_map={0: "#3B82F6", 1: "#EF4444"},
            labels={"time_delta_mins": "Minutes Since Prior Txn", "amount": "Amount ($)"},
            title="Time Gap vs Transaction Amount"
        )
        fig_vel.update_layout(height=320)
        st.plotly_chart(fig_vel, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: MODEL INTELLIGENCE & FEATURE IMPORTANCE
# ---------------------------------------------------------
with tab_model:
    st.subheader("🧠 Machine Learning & Statistical Insights")
    st.write(f"Active Algorithm: **{algo_choice}** | Contamination Setting: **{contamination_rate:.1%}**")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("#### Feature Correlation Heatmap")
        corr_features = ["amount", "amount_to_mean_ratio", "amount_zscore", "time_delta_mins", "txns_last_1h", "spend_last_24h", "anomaly_score"]
        corr_matrix = df_results[corr_features].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix of Anomaly Risk Indicators"
        )
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

    with m_col2:
        st.markdown("#### 2D Feature Space: Amount Z-Score vs Velocity")
        fig_scatter_2d = px.scatter(
            df_results.sample(min(1500, len(df_results)), random_state=42),
            x="amount_zscore",
            y="txns_last_1h",
            color="anomaly_score",
            size="amount",
            color_continuous_scale="Turbo",
            title="Amount Deviation (Z-Score) vs 1-Hour Velocity",
            labels={"amount_zscore": "Amount Z-Score (Deviations from Mean)", "txns_last_1h": "Txns in Last 1 Hour"}
        )
        fig_scatter_2d.update_layout(height=420)
        st.plotly_chart(fig_scatter_2d, use_container_width=True)

    if ground_truth_metrics:
        st.markdown("---")
        st.subheader("🎯 Ground Truth Performance Evaluation")
        v1, v2, v3 = st.columns(3)
        v1.metric("Precision", f"{ground_truth_metrics['precision'] * 100:.1f}%", help="Percentage of flagged alerts that are genuine anomalies.")
        v2.metric("Recall", f"{ground_truth_metrics['recall'] * 100:.1f}%", help="Percentage of all true anomalies successfully caught.")
        v3.metric("F1-Score", f"{ground_truth_metrics['f1_score'] * 100:.1f}%", help="Harmonic mean of precision and recall.")

# ---------------------------------------------------------
# TAB 5: AUDIT REPORT & EXPORT
# ---------------------------------------------------------
with tab_report:
    st.subheader("📋 Compliance & Fraud Audit Report")
    st.markdown("Generate and export full audit records for flagged transactions and risk analysis.")

    flagged_export = df_results[df_results["is_detected_anomaly"] == 1].copy()
    flagged_export.sort_values(by="anomaly_score", ascending=False, inplace=True)

    st.markdown(f"""
    ### Executive Audit Summary
    - **Total Analyzed Records:** {len(df_results):,} transactions
    - **Identified Anomalies:** {len(flagged_export):,} ({len(flagged_export)/len(df_results):.2%})
    - **Total Flagged Capital at Risk:** **${flagged_export['amount'].sum():,.2f}**
    - **Detection Model:** {algo_choice} (Contamination: {contamination_rate:.1%})
    - **Alert Score Threshold:** {custom_threshold} / 100
    - **Report Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """)

    st.markdown("#### High-Risk Customer Accounts (Ranked by Anomaly Frequency)")
    risk_cust = flagged_export.groupby("customer_id").agg(
        anomaly_count=("transaction_id", "count"),
        total_risk_amount=("amount", "sum"),
        avg_score=("anomaly_score", "mean")
    ).reset_index().sort_values(by="anomaly_count", ascending=False).head(10)
    st.dataframe(risk_cust, use_container_width=True)

    export_cols = [
        "transaction_id", "customer_id", "timestamp", "amount",
        "anomaly_score", "risk_level", "transaction_type",
        "merchant_category", "channel", "location", "anomaly_reasons"
    ]
    csv_data = flagged_export[export_cols].to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Detailed Anomaly Audit Report (CSV)",
        data=csv_data,
        file_name=f"bank_anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )
