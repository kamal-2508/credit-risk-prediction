# ─────────────────────────────────────────────────────────────────────────────
# app/statistical_analysis.py
#
# WHY THIS FILE EXISTS:
#   Adds rigorous statistical and probabilistic analysis to the credit risk
#   project. This demonstrates:
#   1. Non-Gaussian distribution modelling
#   2. Statistical hypothesis testing
#   3. Distribution fitting
#   4. Probability calibration
#   5. CDF-based evaluation (KS statistic)
#
# Directly relevant to Cadence JD:
#   "modelling mathematical operations on non-Gaussian probability
#    distribution models"
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    kstest, mannwhitneyu, shapiro,
    beta, gamma, norm, expon,
    ks_2samp
)
import logging

logger = logging.getLogger(__name__)


# ─── 1. Normality Testing ─────────────────────────────────────────────────────

def test_normality(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Test whether features follow a Gaussian distribution.

    WHY: Cadence needs non-Gaussian modelling. First we prove
    the features ARE non-Gaussian using Shapiro-Wilk test,
    then justify using distribution-free methods (KS statistic).

    Shapiro-Wilk: H0 = data is normally distributed
    p < 0.05 → reject H0 → non-Gaussian confirmed
    """
    results = []
    for feat in features:
        if feat not in df.columns:
            continue
        data = df[feat].dropna().values

        # Shapiro-Wilk (best for n < 5000)
        if len(data) <= 5000:
            stat, p = shapiro(data)
            test_used = "Shapiro-Wilk"
        else:
            # Use KS test against normal for larger samples
            stat, p = kstest(
                (data - data.mean()) / data.std(),
                "norm"
            )
            test_used = "KS vs Normal"

        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        results.append({
            "feature":    feat,
            "test":       test_used,
            "statistic":  round(float(stat), 4),
            "p_value":    round(float(p), 4),
            "is_gaussian": p > 0.05,
            "skewness":   round(float(skewness), 4),
            "kurtosis":   round(float(kurtosis), 4),
            "verdict":    "Gaussian" if p > 0.05 else "Non-Gaussian",
        })

    df_results = pd.DataFrame(results)
    n_nongaussian = (df_results["is_gaussian"] == False).sum()
    logger.info(f"✓ Normality tests: {n_nongaussian}/{len(features)} features are Non-Gaussian")
    return df_results


# ─── 2. Two-Sample KS Test (Defaulters vs Non-Defaulters) ─────────────────────

def ks_separation_analysis(df: pd.DataFrame, features: list, target_col: str = "target") -> pd.DataFrame:
    """
    Apply two-sample KS test between defaulters and non-defaulters
    for each feature.

    WHY: The KS statistic measures the maximum distance between
    two empirical CDFs — a distribution-free (non-Gaussian) method.
    High KS = feature strongly separates the two populations.

    This is the core probabilistic insight for Cadence:
    we model the DISTRIBUTIONAL DIFFERENCE not just mean difference.
    """
    defaulters     = df[df[target_col] == 1]
    non_defaulters = df[df[target_col] == 0]

    results = []
    for feat in features:
        if feat not in df.columns:
            continue

        d_vals  = defaulters[feat].dropna().values
        nd_vals = non_defaulters[feat].dropna().values

        if len(d_vals) < 5 or len(nd_vals) < 5:
            continue

        ks_stat, p_val = ks_2samp(d_vals, nd_vals)

        # Mann-Whitney U test (non-parametric alternative to t-test)
        mw_stat, mw_p = mannwhitneyu(d_vals, nd_vals, alternative="two-sided")

        results.append({
            "feature":         feat,
            "ks_statistic":    round(float(ks_stat), 4),
            "ks_p_value":      round(float(p_val), 4),
            "mw_p_value":      round(float(mw_p), 4),
            "significant":     p_val < 0.05,
            "mean_default":    round(float(d_vals.mean()), 4),
            "mean_no_default": round(float(nd_vals.mean()), 4),
            "separation":      "Strong" if ks_stat > 0.3 else "Moderate" if ks_stat > 0.15 else "Weak",
        })

    df_results = pd.DataFrame(results).sort_values("ks_statistic", ascending=False)
    logger.info(f"✓ KS separation analysis complete for {len(results)} features")
    return df_results


# ─── 3. Distribution Fitting ──────────────────────────────────────────────────

def fit_distributions(data: np.ndarray, feature_name: str) -> dict:
    """
    Fit multiple probability distributions to a feature and find best fit.

    WHY: Non-Gaussian modelling requires identifying WHICH distribution
    fits best — Beta, Gamma, Exponential etc. This is the core
    mathematical operation Cadence is looking for.

    Distributions tested:
    - Normal (Gaussian baseline)
    - Beta (bounded 0-1, good for ratios)
    - Gamma (right-skewed, good for amounts)
    - Exponential (good for time/duration data)
    """
    distributions = {
        "normal":      norm,
        "gamma":       gamma,
        "exponential": expon,
    }

    # Normalize data to [0,1] for Beta distribution
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    distributions["beta"] = beta

    results = {}
    best_dist = None
    best_aic  = np.inf

    for name, dist in distributions.items():
        try:
            d = data_norm if name == "beta" else data

            # Fit distribution parameters using MLE
            params = dist.fit(d)

            # Compute log-likelihood
            log_likelihood = np.sum(dist.logpdf(d, *params))

            # AIC = 2k - 2*log_likelihood (lower is better)
            k   = len(params)
            aic = 2 * k - 2 * log_likelihood

            # KS goodness of fit test
            ks_stat, ks_p = kstest(d, dist.name, args=params)

            results[name] = {
                "params":         params,
                "aic":            round(float(aic), 2),
                "log_likelihood": round(float(log_likelihood), 2),
                "ks_statistic":   round(float(ks_stat), 4),
                "ks_p_value":     round(float(ks_p), 4),
                "good_fit":       ks_p > 0.05,
            }

            if aic < best_aic:
                best_aic  = aic
                best_dist = name

        except Exception as e:
            results[name] = {"error": str(e)}

    results["best_fit"] = best_dist
    results["feature"]  = feature_name
    logger.info(f"  {feature_name}: best fit = {best_dist} (AIC={best_aic:.2f})")
    return results


def fit_all_features(df: pd.DataFrame, num_features: list) -> dict:
    """Fit distributions for all numerical features."""
    logger.info("Fitting probability distributions to numerical features...")
    all_fits = {}
    for feat in num_features:
        if feat in df.columns:
            data = df[feat].dropna().values
            if len(data) > 20:
                all_fits[feat] = fit_distributions(data, feat)
    return all_fits


# ─── 4. Default Probability Calibration ──────────────────────────────────────

def calibrate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Analyse the calibration of model probability outputs.

    WHY: A well-calibrated model means P(default=1|score=0.7) = 0.70
    This is critical in credit risk — the raw score must be a true
    probability, not just a ranking score.

    We model the calibration error as a Beta distribution problem:
    predicted probabilities should follow Beta(α, β) distribution.
    """
    from sklearn.calibration import calibration_curve

    # Compute calibration curve
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

    # Expected Calibration Error (ECE)
    ece = float(np.mean(np.abs(fraction_pos - mean_pred)))

    # Fit Beta distribution to predicted probabilities
    try:
        beta_params = beta.fit(y_prob, floc=0, fscale=1)
        alpha, beta_param = beta_params[0], beta_params[1]
        dist_shape = "right-skewed" if alpha < beta_param else "left-skewed"
    except Exception:
        alpha, beta_param, dist_shape = None, None, "unknown"

    # Brier Score (mean squared error of probabilities)
    brier = float(np.mean((y_prob - y_true) ** 2))

    result = {
        "ece":              round(ece, 4),
        "brier_score":      round(brier, 4),
        "beta_alpha":       round(float(alpha), 4) if alpha else None,
        "beta_param":       round(float(beta_param), 4) if beta_param else None,
        "distribution_shape": dist_shape,
        "calibration_quality": "Good" if ece < 0.05 else "Moderate" if ece < 0.10 else "Poor",
        "fraction_pos":     fraction_pos.tolist(),
        "mean_pred":        mean_pred.tolist(),
    }

    logger.info(f"✓ Calibration: ECE={ece:.4f} Brier={brier:.4f} ({result['calibration_quality']})")
    return result


# ─── 5. Full Statistical Report ───────────────────────────────────────────────

def run_full_statistical_analysis(df: pd.DataFrame) -> dict:
    """
    Run complete statistical analysis pipeline.
    Call this after loading processed data.
    """
    logger.info("=" * 50)
    logger.info("STATISTICAL ANALYSIS PIPELINE")
    logger.info("=" * 50)

    num_features = [
        "duration", "credit_amount", "installment_rate",
        "age", "existing_credits", "dependents",
        "monthly_repayment", "installment_burden", "credit_per_age"
    ]
    num_features = [f for f in num_features if f in df.columns]

    # 1. Normality tests
    logger.info("\n1. Testing for Gaussian distribution...")
    normality = test_normality(df, num_features)

    # 2. KS separation between defaulters and non-defaulters
    logger.info("\n2. KS separation analysis...")
    separation = ks_separation_analysis(df, num_features)

    # 3. Distribution fitting
    logger.info("\n3. Fitting probability distributions...")
    dist_fits = fit_all_features(df, num_features)

    return {
        "normality_tests":   normality,
        "ks_separation":     separation,
        "distribution_fits": dist_fits,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.load_data import load_processed_data
    from app.features import engineer_features

    df = load_processed_data()
    df = engineer_features(df)

    results = run_full_statistical_analysis(df)

    print("\n📊 Normality Tests:")
    print(results["normality_tests"][["feature", "verdict", "skewness", "kurtosis"]].to_string(index=False))

    print("\n📊 KS Separation (top 5):")
    print(results["ks_separation"].head(5)[["feature", "ks_statistic", "separation"]].to_string(index=False))

    print("\n📊 Best Distribution Fits:")
    for feat, fit in list(results["distribution_fits"].items())[:5]:
        print(f"  {feat}: {fit.get('best_fit', 'N/A')}")
