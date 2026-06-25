# ─────────────────────────────────────────────────────────────────────────────
# app/features.py
#
# Feature engineering for credit risk prediction.
# Creates domain-specific features that improve model performance.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific features for credit risk.

    WHY FEATURE ENGINEERING MATTERS:
    Raw features like 'duration' and 'credit_amount' are useful alone,
    but combining them reveals patterns — a ₹50,000 loan over 6 months
    is very different from the same amount over 60 months.
    """
    df = df.copy()
    logger.info("Engineering features...")

    # ── Ratio features ────────────────────────────────────────────────────────

    # Monthly repayment burden (higher = riskier)
    if "credit_amount" in df.columns and "duration" in df.columns:
        df["monthly_repayment"] = df["credit_amount"] / df["duration"].clip(lower=1)

    # Credit amount per year of employment (proxy for affordability)
    if "credit_amount" in df.columns and "age" in df.columns:
        df["credit_per_age"] = df["credit_amount"] / df["age"].clip(lower=1)

    # Installment burden score
    if "installment_rate" in df.columns and "duration" in df.columns:
        df["installment_burden"] = df["installment_rate"] * df["duration"]

    # ── Risk flag features ────────────────────────────────────────────────────

    # High credit amount flag (top 25%)
    if "credit_amount" in df.columns:
        threshold = df["credit_amount"].quantile(0.75)
        df["high_credit_flag"] = (df["credit_amount"] > threshold).astype(int)

    # Long duration flag (> 2 years)
    if "duration" in df.columns:
        df["long_duration_flag"] = (df["duration"] > 24).astype(int)

    # Young borrower flag (< 25 years) — statistically higher risk
    if "age" in df.columns:
        df["young_borrower_flag"] = (df["age"] < 25).astype(int)
        df["senior_borrower_flag"] = (df["age"] > 50).astype(int)

    # Multiple existing credits flag
    if "existing_credits" in df.columns:
        df["multiple_credits_flag"] = (df["existing_credits"] > 1).astype(int)

    logger.info(f"✓ Engineered {df.shape[1]} total features")
    return df


def get_feature_names(df: pd.DataFrame) -> list:
    """Return all feature column names (excluding target)."""
    return [c for c in df.columns if c != "target"]
