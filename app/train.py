# ─────────────────────────────────────────────────────────────────────────────
# app/train.py
#
# Trains and evaluates ensemble models for credit risk prediction.
# Models: Logistic Regression, Random Forest, XGBoost, LightGBM, Stacking
#
# Run: python app/train.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.load_data import load_processed_data
from app.features import engineer_features, get_feature_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ─── Data preparation ─────────────────────────────────────────────────────────

def prepare_data():
    """Load, engineer features, and split data."""
    logger.info("Loading data...")
    df = load_processed_data()
    df = engineer_features(df)

    feature_cols = get_feature_names(df)
    X = df[feature_cols]
    y = df["target"]

    logger.info(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
    logger.info(f"Default rate: {y.mean():.1%} ({y.sum()} defaults out of {len(y)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols


# ─── Model definitions ────────────────────────────────────────────────────────

def get_models():
    """
    Define all models to train.

    WHY THESE MODELS:
    - Logistic Regression: baseline, interpretable, fast
    - Random Forest: handles non-linearity, robust to outliers
    - GradientBoosting: strong performer on tabular data
    - XGBoost: industry standard for credit scoring
    - LightGBM: faster XGBoost, great on imbalanced data
    - Stacking: combines all models for best performance
    """
    models = {}

    # Baseline
    models["logistic_regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=0.1,
        ))
    ])

    # Random Forest
    models["random_forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Gradient Boosting (sklearn — no extra install needed)
    models["gradient_boosting"] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    # Try XGBoost if available
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.3,  # handles class imbalance
            random_state=RANDOM_STATE,
            eval_metric="auc",
            verbosity=0,
        )
        logger.info("✓ XGBoost available")
    except ImportError:
        logger.warning("XGBoost not installed, skipping")

    # Try LightGBM if available
    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        logger.info("✓ LightGBM available")
    except ImportError:
        logger.warning("LightGBM not installed, skipping")

    return models


def build_stacking_ensemble(base_models: dict):
    """
    Build a stacking ensemble using trained base models as estimators.
    Meta-learner: Logistic Regression (keeps it interpretable).
    """
    # Use RF + GB as base (always available), add XGB/LGBM if available
    estimators = [
        ("rf",  base_models["random_forest"]),
        ("gb",  base_models["gradient_boosting"]),
    ]
    if "xgboost" in base_models:
        estimators.append(("xgb", base_models["xgboost"]))
    if "lightgbm" in base_models:
        estimators.append(("lgbm", base_models["lightgbm"]))

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=RANDOM_STATE, C=0.1),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    return stack


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a trained model and return metrics dict."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc     = roc_auc_score(y_test, y_prob)
    ap      = average_precision_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)

    # KS Statistic — key metric in credit scoring
    # Measures max separation between default and non-default score distributions
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ks_stat = max(tpr - fpr)

    # Gini coefficient (common in credit risk)
    gini = 2 * auc - 1

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics = {
        "model":     model_name,
        "auc_roc":   round(auc,       4),
        "gini":      round(gini,       4),
        "ks_stat":   round(ks_stat,    4),
        "avg_prec":  round(ap,         4),
        "precision": round(precision,  4),
        "recall":    round(recall,     4),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }

    logger.info(
        f"  {model_name:<25} AUC={auc:.4f}  KS={ks_stat:.4f}  Gini={gini:.4f}"
    )
    return metrics


def cross_validate_model(model, X_train, y_train, model_name: str) -> float:
    """5-fold CV AUC on training set."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info(f"  {model_name:<25} CV AUC={scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


# ─── Feature importance ───────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            if hasattr(clf, "coef_"):
                imp = np.abs(clf.coef_[0])
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        df_imp = pd.DataFrame({
            "feature":    feature_names,
            "importance": imp,
            "model":      model_name,
        }).sort_values("importance", ascending=False)

        return df_imp

    except Exception:
        return pd.DataFrame()


# ─── SHAP explainability ──────────────────────────────────────────────────────

def compute_shap_values(model, X_test: pd.DataFrame, model_name: str):
    """Compute SHAP values if shap is installed."""
    try:
        import shap

        if "random_forest" in model_name or "gradient" in model_name:
            explainer = shap.TreeExplainer(model)
        elif "xgboost" in model_name or "lightgbm" in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_test)

        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (default)

        path = os.path.join(MODELS_DIR, f"shap_{model_name}.npy")
        np.save(path, shap_values)
        logger.info(f"  ✓ SHAP values saved → {path}")
        return shap_values

    except ImportError:
        logger.warning("SHAP not installed — run: pip install shap")
        return None
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


# ─── Main training pipeline ───────────────────────────────────────────────────

def train_all():
    """Full training pipeline."""
    logger.info("=" * 60)
    logger.info("CREDIT RISK PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data()

    # 2. Get all models
    models = get_models()

    # 3. Cross-validate
    logger.info("\n📊 Cross-validation (5-fold AUC on training set):")
    cv_results = {}
    for name, model in models.items():
        cv_results[name] = cross_validate_model(model, X_train, y_train, name)

    # 4. Train all models
    logger.info("\n🏋️  Training models...")
    trained = {}
    for name, model in models.items():
        logger.info(f"  Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))

    # 5. Build and train stacking ensemble
    logger.info("  Training stacking ensemble...")
    stack = build_stacking_ensemble(trained)
    stack.fit(X_train, y_train)
    trained["stacking_ensemble"] = stack
    joblib.dump(stack, os.path.join(MODELS_DIR, "stacking_ensemble.pkl"))

    # 6. Evaluate all on test set
    logger.info("\n📈 Test set evaluation:")
    all_metrics = []
    for name, model in trained.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)

    # 7. Save metrics
    metrics_df = pd.DataFrame(all_metrics).sort_values("auc_roc", ascending=False)
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    metrics_df.to_json(metrics_path, orient="records", indent=2)
    logger.info(f"\n✓ Metrics saved → {metrics_path}")

    # 8. Feature importance for best model
    best_name = metrics_df.iloc[0]["model"]
    best_model = trained[best_name]
    logger.info(f"\n🏆 Best model: {best_name} (AUC={metrics_df.iloc[0]['auc_roc']:.4f})")

    # Save feature importance
    imp_df = get_feature_importance(best_model, feature_cols, best_name)
    if not imp_df.empty:
        imp_path = os.path.join(MODELS_DIR, "feature_importance.csv")
        imp_df.to_csv(imp_path, index=False)
        logger.info(f"✓ Feature importance saved → {imp_path}")
        logger.info("\nTop 10 features:")
        print(imp_df.head(10).to_string(index=False))

    # 9. SHAP values for best tree model
    best_tree = trained.get("random_forest") or trained.get("gradient_boosting")
    if best_tree:
        logger.info("\n🔍 Computing SHAP values...")
        compute_shap_values(best_tree, X_test, "random_forest")

    # 10. Save feature columns for inference
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    print(metrics_df[["model", "auc_roc", "gini", "ks_stat", "precision", "recall"]].to_string(index=False))

    return trained, metrics_df


if __name__ == "__main__":
    train_all()
