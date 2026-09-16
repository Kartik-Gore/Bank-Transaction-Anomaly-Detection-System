"""
tests/test_anomaly_detection.py
Unit test suite verifying data generation, feature engineering calculations,
multi-model anomaly scoring, and reason code generation.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add src to sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_generator import generate_bank_transactions
from feature_engineering import extract_features
from anomaly_detector import BankAnomalyDetector

class TestBankAnomalyDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate small dataset for testing without overwriting production data
        cls.raw_df = generate_bank_transactions(
            n_transactions=300,
            n_customers=25,
            anomaly_rate=0.08,
            random_state=123,
            save_path=os.path.join(project_root, "data", "test_transactions.csv")
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up test transactions file
        test_file = os.path.join(project_root, "data", "test_transactions.csv")
        if os.path.exists(test_file):
            try:
                os.remove(test_file)
            except:
                pass

    def test_01_data_generator_schema(self):
        """Test that data generator creates all expected columns and valid values."""
        expected_cols = [
            "transaction_id", "customer_id", "timestamp", "amount",
            "transaction_type", "merchant_category", "channel", "location",
            "is_anomaly", "anomaly_type"
        ]
        for col in expected_cols:
            self.assertIn(col, self.raw_df.columns, f"Missing expected column: {col}")

        self.assertEqual(len(self.raw_df), 300)
        self.assertTrue((self.raw_df["amount"] > 0).all(), "All transaction amounts must be positive")
        self.assertGreater(self.raw_df["is_anomaly"].sum(), 0, "Should contain injected anomalies")

    def test_02_feature_engineering(self):
        """Test feature extraction and statistical profiling calculations."""
        enriched, feature_cols = extract_features(self.raw_df)

        self.assertEqual(len(enriched), len(self.raw_df))
        self.assertGreaterEqual(len(feature_cols), 10)

        key_features = ["amount_zscore", "amount_to_mean_ratio", "txns_last_1h", "time_delta_mins", "is_night"]
        for kf in key_features:
            self.assertIn(kf, feature_cols)

        for col in feature_cols:
            self.assertFalse(enriched[col].isna().any(), f"NaN values detected in feature: {col}")
            self.assertFalse(np.isinf(enriched[col]).any(), f"Inf values detected in feature: {col}")

    def test_03_isolation_forest_detector(self):
        """Test Isolation Forest model fitting, scoring range, and risk tiers."""
        enriched, feature_cols = extract_features(self.raw_df)
        detector = BankAnomalyDetector(algorithm="Isolation Forest", contamination=0.08, random_state=42)
        results = detector.fit_predict(enriched, feature_cols)

        self.assertIn("anomaly_score", results.columns)
        self.assertIn("risk_level", results.columns)
        self.assertIn("anomaly_reasons", results.columns)

        scores = results["anomaly_score"]
        self.assertTrue((scores >= 0.0).all() and (scores <= 100.0).all(), "Scores must be strictly bounded in [0, 100]")

        valid_tiers = {"Low Risk", "Moderate Risk", "Elevated Risk", "Critical Risk"}
        for tier in results["risk_level"].unique():
            self.assertIn(tier, valid_tiers)

        for reason in results["anomaly_reasons"]:
            self.assertIsInstance(reason, str)
            self.assertGreater(len(reason), 0)

    def test_04_local_outlier_factor_detector(self):
        """Test Local Outlier Factor algorithm runs without error."""
        enriched, feature_cols = extract_features(self.raw_df)
        detector = BankAnomalyDetector(algorithm="Local Outlier Factor", contamination=0.08, random_state=42)
        results = detector.fit_predict(enriched, feature_cols)

        self.assertIn("anomaly_score", results.columns)
        self.assertIn("is_detected_anomaly", results.columns)
        self.assertGreaterEqual(results["is_detected_anomaly"].sum(), 1)

    def test_05_performance_evaluation(self):
        """Test metric calculation against ground truth labels."""
        enriched, feature_cols = extract_features(self.raw_df)
        detector = BankAnomalyDetector(algorithm="Isolation Forest", contamination=0.08, random_state=42)
        results = detector.fit_predict(enriched, feature_cols)
        metrics = detector.evaluate_performance(results)

        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertTrue(0.0 <= metrics["precision"] <= 1.0)
        self.assertTrue(0.0 <= metrics["recall"] <= 1.0)

if __name__ == "__main__":
    unittest.main()
