# 🛡️ Bank-Transaction-Anomaly-Detection

A Python-based banking transaction analysis and fraud anomaly detection system powered by **Scikit-Learn (Isolation Forest & Local Outlier Factor)**, **Pandas**, **NumPy**, and an interactive **Streamlit Web Dashboard**.

---

## 📁 Project Directory Structure

```
Bank-Transaction-Anomaly-Detection/
│
├── data/
│   ├── transactions.csv          # Raw / synthesized transaction records
│   └── anomaly_report.csv        # Exported audit report of flagged anomalies
│
├── src/
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # Standalone CLI execution pipeline
│   ├── anomaly_detector.py       # ML models (Isolation Forest, LOF) & scoring engine
│   ├── feature_engineering.py    # Customer behavioral baselines, velocity & z-scores
│   └── data_generator.py         # Realistic bank dataset generator with fraud patterns
│
├── dashboard/
│   └── app.py                    # Streamlit interactive web dashboard
│
├── tests/
│   └── test_anomaly_detection.py # Automated unit test suite
│
├── outputs/
│   └── anomaly_analysis.png      # 4-panel visual summary charts
│
├── README.md                     # Project guide & documentation
├── requirements.txt              # Project dependencies
└── .gitignore                    # Git ignore configurations
```

---

## 🌟 Key Features

1. **Transaction Analysis & Customer Profiling**:
   - Calculates customer spending baselines (mean, median, standard deviation).
   - Computes amount deviation metrics: $z$-scores and amount-to-mean ratios.
   - Computes velocity and frequency metrics: rolling counts in 1-hour and 24-hour windows, and time elapsed since prior transaction.
   - Encodes temporal indicators (hour of day, night-time transfer flags, weekend flags) and high-risk merchant categories.

2. **Multi-Model Anomaly Detection**:
   - Powered by Scikit-Learn **Isolation Forest** and **Local Outlier Factor (LOF)**.
   - Produces a calibrated **Anomaly Score (0–100)** categorized into:
     - 🟢 *Low Risk (0–40)*
     - 🟡 *Moderate Risk (40–50)*
     - 🟠 *Elevated Risk (50–75)*
     - 🔴 *Critical Risk (75–100)*
   - Generates plain-English **explainability reason codes** for every flagged transaction.

3. **Interactive Streamlit Web Dashboard**:
   - **Executive Dashboard**: Key metrics cards, transaction stream timeline, risk donut chart, and score distribution.
   - **Transaction Explorer**: Search and filter by risk tier/category with an interactive **Behavioral Autopsy** inspector.
   - **Customer Profiling**: Look up any customer's baseline, spending history, and category breakdown.
   - **Model Intelligence**: Feature correlation heatmap and 2D feature projection.
   - **Audit Report**: Compliance audit view with one-click **CSV report download**.

4. **Automated CLI Pipeline & Visual Reports**:
   - Headless execution mode generates `data/anomaly_report.csv` and a 4-panel publication-quality Matplotlib/Seaborn visualization (`outputs/anomaly_analysis.png`).

---

## 🚀 Quickstart & Execution Commands

### 1. Installation
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard
Run from the root directory:
```bash
streamlit run dashboard/app.py
```
*Opens the web dashboard in your browser (typically `http://localhost:8501`).*

### 3. Run the Standalone CLI Pipeline
To run the automated analysis, export `data/anomaly_report.csv`, and produce `outputs/anomaly_analysis.png`:
```bash
python src/main.py
```

### 4. Run Automated Unit Tests
To verify all pipelines, feature calculations, and model predictions:
```bash
python -m unittest discover tests
```
*(or `python tests/test_anomaly_detection.py`)*
