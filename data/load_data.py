# ─────────────────────────────────────────────────────────────────────────────
# data/load_data.py
#
# Downloads the German Credit Risk dataset from UCI repository.
# No Kaggle account needed — downloads automatically.
#
# Dataset: 1000 loan applicants, 20 features, binary target (good/bad credit)
# Source: UCI Machine Learning Repository
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR  = os.path.dirname(__file__)
RAW_PATH  = os.path.join(DATA_DIR, "german_credit_raw.csv")
PROC_PATH = os.path.join(DATA_DIR, "german_credit_processed.csv")

# Column names for the UCI German Credit dataset
COLUMN_NAMES = [
    "checking_account",       # Status of existing checking account
    "duration",               # Duration in months
    "credit_history",         # Credit history
    "purpose",                # Purpose of loan
    "credit_amount",          # Credit amount
    "savings_account",        # Savings account/bonds
    "employment_since",       # Present employment since
    "installment_rate",       # Installment rate in % of disposable income
    "personal_status",        # Personal status and sex
    "other_debtors",          # Other debtors / guarantors
    "residence_since",        # Present residence since
    "property",               # Property
    "age",                    # Age in years
    "other_installments",     # Other installment plans
    "housing",                # Housing
    "existing_credits",       # Number of existing credits at this bank
    "job",                    # Job
    "dependents",             # Number of people being liable to provide maintenance
    "telephone",              # Telephone
    "foreign_worker",         # Foreign worker
    "target",                 # 1 = good credit, 2 = bad credit
]


def download_data() -> pd.DataFrame:
    """Download German Credit dataset from UCI if not already present."""
    if os.path.exists(RAW_PATH):
        logger.info("✓ Raw data already exists, loading from disk")
        return pd.read_csv(RAW_PATH, sep=" ", header=None, names=COLUMN_NAMES)

    logger.info("Downloading German Credit dataset from UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    try:
        df = pd.read_csv(url, sep=" ", header=None, names=COLUMN_NAMES)
        df.to_csv(RAW_PATH, index=False)
        logger.info(f"✓ Downloaded {len(df)} records → {RAW_PATH}")
        return df
    except Exception as e:
        logger.warning(f"UCI download failed ({e}), generating synthetic data instead")
        return generate_synthetic_data()


def generate_synthetic_data(n=1000, seed=42) -> pd.DataFrame:
    """
    Generate realistic synthetic credit risk data as fallback.
    Same schema as German Credit dataset.
    """
    np.random.seed(seed)
    logger.info(f"Generating {n} synthetic credit records...")

    checking = np.random.choice(["A11", "A12", "A13", "A14"], n, p=[0.27, 0.27, 0.06, 0.40])
    duration = np.random.randint(4, 72, n)
    credit_amount = np.random.randint(250, 18000, n)
    age = np.random.randint(19, 75, n)
    installment_rate = np.random.randint(1, 5, n)
    employment = np.random.choice(["A71", "A72", "A73", "A74", "A75"], n)
    savings = np.random.choice(["A61", "A62", "A63", "A64", "A65"], n, p=[0.60, 0.10, 0.06, 0.06, 0.18])
    purpose = np.random.choice(["A40", "A41", "A42", "A43", "A44", "A45"], n)
    credit_history = np.random.choice(["A30", "A31", "A32", "A33", "A34"], n)
    housing = np.random.choice(["A151", "A152", "A153"], n, p=[0.18, 0.71, 0.11])

    # Generate target with realistic correlation to features
    risk_score = (
        (checking == "A14").astype(int) * 0.3 +
        (duration < 12).astype(int) * 0.2 +
        (credit_amount < 2000).astype(int) * 0.15 +
        (age > 35).astype(int) * 0.15 +
        (savings == "A65").astype(int) * 0.2 +
        np.random.uniform(0, 0.3, n)
    )
    target = np.where(risk_score > 0.45, 1, 2)  # 1=good, 2=bad (UCI convention)

    df = pd.DataFrame({
        "checking_account":   checking,
        "duration":           duration,
        "credit_history":     credit_history,
        "purpose":            purpose,
        "credit_amount":      credit_amount,
        "savings_account":    savings,
        "employment_since":   employment,
        "installment_rate":   installment_rate,
        "personal_status":    np.random.choice(["A91", "A92", "A93", "A94"], n),
        "other_debtors":      np.random.choice(["A101", "A102", "A103"], n),
        "residence_since":    np.random.randint(1, 5, n),
        "property":           np.random.choice(["A121", "A122", "A123", "A124"], n),
        "age":                age,
        "other_installments": np.random.choice(["A141", "A142", "A143"], n),
        "housing":            housing,
        "existing_credits":   np.random.randint(1, 5, n),
        "job":                np.random.choice(["A171", "A172", "A173", "A174"], n),
        "dependents":         np.random.randint(1, 3, n),
        "telephone":          np.random.choice(["A191", "A192"], n),
        "foreign_worker":     np.random.choice(["A201", "A202"], n, p=[0.96, 0.04]),
        "target":             target,
    })

    df.to_csv(RAW_PATH, index=False)
    logger.info(f"✓ Generated synthetic data → {RAW_PATH}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw German Credit data into model-ready format.

    Steps:
    1. Convert target: 1=good → 0 (no default), 2=bad → 1 (default)
    2. One-hot encode categorical features
    3. Keep numerical features as-is
    """
    logger.info("Preprocessing data...")

    df = df.copy()

    # Convert target: UCI uses 1=good, 2=bad → we want 0=good, 1=default
    df["target"] = (df["target"] == 2).astype(int)

    # Categorical columns to encode
    cat_cols = [
        "checking_account", "credit_history", "purpose", "savings_account",
        "employment_since", "personal_status", "other_debtors", "property",
        "other_installments", "housing", "job", "telephone", "foreign_worker"
    ]

    # Numerical columns to keep
    num_cols = [
        "duration", "credit_amount", "installment_rate", "residence_since",
        "age", "existing_credits", "dependents"
    ]

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)

    # Combine with numerical
    df_final = pd.concat([df[num_cols], df_encoded, df["target"]], axis=1)

    df_final.to_csv(PROC_PATH, index=False)
    logger.info(f"✓ Preprocessed: {df_final.shape[0]} rows × {df_final.shape[1]} cols → {PROC_PATH}")

    return df_final


def load_processed_data() -> pd.DataFrame:
    """Main entry point — returns processed data ready for modeling."""
    if os.path.exists(PROC_PATH):
        logger.info("✓ Loading preprocessed data from disk")
        return pd.read_csv(PROC_PATH)

    raw = download_data()
    return preprocess_data(raw)


if __name__ == "__main__":
    df = load_processed_data()
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"   Default rate:  {df['target'].mean():.1%}")
    print(f"   Features:      {df.shape[1] - 1}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
