"""
src/anomaly_detector.py
Multi-model anomaly detection engine using Scikit-learn (Isolation Forest & Local Outlier Factor),
statistical thresholding, explainable scoring (0-100), and automated reason code generation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class BankAnomalyDetector:
    def __init__(
        self,
        algorithm: str = "Isolation Forest",
        contamination: float = 0.05,
        random_state: int = 42
    ):
        self.algorithm = algorithm
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.model = None
        self.feature_cols = []

    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        self.feature_cols = feature_cols
        res_df = df.copy()

        X = res_df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale features using RobustScaler (outlier resilient)
        X_scaled = self.scaler.fit_transform(X)

        if self.algorithm == "Isolation Forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=150,
                max_samples="auto",
                n_jobs=-1
            )
            self.model.fit(X_scaled)
            raw_scores = self.model.decision_function(X_scaled)
            min_s, max_s = raw_scores.min(), raw_scores.max()
            if max_s > min_s:
                ml_score = 100.0 * (1.0 - (raw_scores - min_s) / (max_s - min_s))
            else:
                ml_score = np.zeros(len(raw_scores))
            raw_preds = self.model.predict(X_scaled)
            is_anomaly_ml = (raw_preds == -1).astype(int)

        elif self.algorithm == "Local Outlier Factor":
            self.model = LocalOutlierFactor(
                n_neighbors=25,
                contamination=self.contamination,
                novelty=False,
                n_jobs=-1
            )
            raw_preds = self.model.fit_predict(X_scaled)
            raw_scores = self.model.negative_outlier_factor_
            min_s, max_s = raw_scores.min(), raw_scores.max()
            if max_s > min_s:
                ml_score = 100.0 * (1.0 - (raw_scores - min_s) / (max_s - min_s))
            else:
                ml_score = np.zeros(len(raw_scores))
            is_anomaly_ml = (raw_preds == -1).astype(int)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Rule-Based Heuristic Risk Boosters
        heuristic_score = np.zeros(len(res_df))
        
        # Rule 1: High Amount Z-Score
        z_scores = res_df["amount_zscore"].values
        heuristic_score += np.clip((z_scores - 3.0) * 12.0, 0, 45)

        # Rule 2: Amount to Mean Ratio > 4x
        ratio = res_df["amount_to_mean_ratio"].values
        heuristic_score += np.clip((ratio - 4.0) * 8.0, 0, 30)

        # Rule 3: Rapid velocity burst
        heuristic_score += res_df["rapid_succession_flag"].values * 15.0

        # Rule 4: High velocity in 1 hour (> 3 txns)
        txns_1h = res_df["txns_last_1h"].values
        heuristic_score += np.clip((txns_1h - 2) * 10.0, 0, 25)

        # Rule 5: High risk merchant category
        heuristic_score += res_df["is_high_risk_category"].values * 20.0

        # Rule 6: Night-time transfer
        is_night_transfer = (res_df["is_night"] == 1) & (res_df["is_transfer_type"] == 1)
        heuristic_score += is_night_transfer.values * 15.0

        # Composite Calibrated Anomaly Score (0 - 100)
        composite_score = 0.60 * ml_score + 0.40 * np.clip(heuristic_score, 0, 100)
        composite_score = np.clip(np.round(composite_score, 1), 0.0, 100.0)

        res_df["anomaly_score"] = composite_score
        
        threshold = float(np.percentile(composite_score, 100 * (1.0 - self.contamination)))
        threshold = max(55.0, threshold)

        res_df["is_detected_anomaly"] = (composite_score >= threshold).astype(int)

        def get_risk_tier(score):
            if score >= 75.0:
                return "Critical Risk"
            elif score >= 50.0:
                return "Elevated Risk"
            elif score >= 30.0:
                return "Moderate Risk"
            else:
                return "Low Risk"

        res_df["risk_level"] = res_df["anomaly_score"].apply(get_risk_tier)

        res_df["anomaly_reasons"] = [
            self._generate_reasons(row) for _, row in res_df.iterrows()
        ]

        return res_df

    def _generate_reasons(self, row: pd.Series) -> str:
        reasons = []
        if row.get("amount_to_mean_ratio", 0) >= 4.0:
            reasons.append(f"Amount {row['amount_to_mean_ratio']:.1f}x higher than customer average (${row.get('cust_mean_amount', 0):.2f})")
            
        if row.get("amount_zscore", 0) >= 3.0:
            reasons.append(f"Statistical deviation: +{row['amount_zscore']:.1f} standard deviations above norm")
            
        if row.get("txns_last_1h", 0) >= 3:
            reasons.append(f"Velocity spike: {int(row['txns_last_1h'])} transactions within 1 hour")
            
        if row.get("time_delta_mins", 9999) < 15.0:
            reasons.append(f"Rapid succession: Only {row['time_delta_mins']:.1f} mins since previous transaction")
            
        if row.get("is_high_risk_category", 0) == 1:
            reasons.append(f"High-risk merchant industry: {row.get('merchant_category', 'Unknown')}")
            
        if row.get("is_night", 0) == 1 and row.get("is_transfer_type", 0) == 1:
            reasons.append(f"Unusual hours: Night-time transfer at {int(row.get('hour_of_day', 0))}:00")

        if not reasons:
            if row.get("anomaly_score", 0) >= 50.0:
                reasons.append("Multi-dimensional feature deviation (ML pattern anomaly)")
            else:
                reasons.append("Normal transaction behavior")

        return " | ".join(reasons)

    def evaluate_performance(self, df: pd.DataFrame) -> dict:
        if "is_anomaly" not in df.columns:
            return {}

        y_true = df["is_anomaly"].values
        y_pred = df["is_detected_anomaly"].values

        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        return {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist()
        }
