# tests/test_credit_risk.py

import pytest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.load_data import generate_synthetic_data, preprocess_data
from app.features import engineer_features, get_feature_names


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return generate_synthetic_data(n=200, seed=42)


@pytest.fixture(scope="module")
def processed_df(raw_df):
    return preprocess_data(raw_df)


@pytest.fixture(scope="module")
def featured_df(processed_df):
    return engineer_features(processed_df)


# ─── Data loading tests ───────────────────────────────────────────────────────

class TestDataLoading:

    def test_synthetic_data_shape(self, raw_df):
        assert raw_df.shape[0] == 200
        assert raw_df.shape[1] == 21   # 20 features + target

    def test_synthetic_data_has_target(self, raw_df):
        assert "target" in raw_df.columns

    def test_target_is_binary_uci(self, raw_df):
        assert set(raw_df["target"].unique()).issubset({1, 2})

    def test_no_null_values_in_synthetic(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_credit_amount_positive(self, raw_df):
        assert (raw_df["credit_amount"] > 0).all()

    def test_age_range(self, raw_df):
        assert raw_df["age"].between(18, 80).all()

    def test_duration_positive(self, raw_df):
        assert (raw_df["duration"] > 0).all()


# ─── Preprocessing tests ──────────────────────────────────────────────────────

class TestPreprocessing:

    def test_target_converted_to_binary(self, processed_df):
        assert set(processed_df["target"].unique()).issubset({0, 1})

    def test_target_1_means_default(self, raw_df, processed_df):
        # UCI 2 = bad → should become 1 in processed
        n_bad_raw  = (raw_df["target"] == 2).sum()
        n_defaults = processed_df["target"].sum()
        assert n_bad_raw == n_defaults

    def test_processed_has_more_columns_than_raw(self, raw_df, processed_df):
        # One-hot encoding creates more columns
        assert processed_df.shape[1] >= raw_df.shape[1]

    def test_no_categorical_columns_remain(self, processed_df):
        cat_cols = processed_df.select_dtypes(include=["object"]).columns
        assert len(cat_cols) == 0

    def test_numerical_cols_preserved(self, processed_df):
        for col in ["duration", "credit_amount", "age"]:
            assert col in processed_df.columns

    def test_default_rate_reasonable(self, processed_df):
        rate = processed_df["target"].mean()
        assert 0.10 <= rate <= 0.50   # realistic default rate


# ─── Feature engineering tests ────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_monthly_repayment_created(self, featured_df):
        assert "monthly_repayment" in featured_df.columns

    def test_monthly_repayment_correct(self, featured_df):
        expected = featured_df["credit_amount"] / featured_df["duration"].clip(lower=1)
        pd.testing.assert_series_equal(
            featured_df["monthly_repayment"].round(4),
            expected.round(4),
            check_names=False,
        )

    def test_high_credit_flag_is_binary(self, featured_df):
        assert set(featured_df["high_credit_flag"].unique()).issubset({0, 1})

    def test_long_duration_flag_correct(self, featured_df):
        expected = (featured_df["duration"] > 24).astype(int)
        pd.testing.assert_series_equal(
            featured_df["long_duration_flag"],
            expected,
            check_names=False,
        )

    def test_young_borrower_flag_correct(self, featured_df):
        expected = (featured_df["age"] < 25).astype(int)
        pd.testing.assert_series_equal(
            featured_df["young_borrower_flag"],
            expected,
            check_names=False,
        )

    def test_feature_count_increases(self, processed_df, featured_df):
        assert featured_df.shape[1] > processed_df.shape[1]

    def test_no_inf_values(self, featured_df):
        numeric = featured_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_no_nan_values(self, featured_df):
        assert featured_df.isnull().sum().sum() == 0

    def test_get_feature_names_excludes_target(self, featured_df):
        cols = get_feature_names(featured_df)
        assert "target" not in cols
        assert len(cols) == featured_df.shape[1] - 1


# ─── Model training smoke tests ───────────────────────────────────────────────

class TestModelSmoke:

    def test_logistic_regression_trains(self, featured_df):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = featured_df.drop(columns=["target"])
        y = featured_df["target"]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42))
        ])
        model.fit(X, y)
        preds = model.predict_proba(X)[:, 1]
        assert len(preds) == len(y)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_random_forest_trains(self, featured_df):
        from sklearn.ensemble import RandomForestClassifier

        X = featured_df.drop(columns=["target"])
        y = featured_df["target"]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_auc_above_random(self, featured_df):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        X = featured_df.drop(columns=["target"])
        y = featured_df["target"]

        model = RandomForestClassifier(n_estimators=20, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
        assert scores.mean() > 0.5   # better than random
