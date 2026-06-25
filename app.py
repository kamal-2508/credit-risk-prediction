import os
if not os.path.exists("models") or not any(f.endswith(".pkl") for f in os.listdir("models")):
    import subprocess
    subprocess.run(["python", "app/train.py"], check=True)

# ─────────────────────────────────────────────────────────────────────────────
# app/main.py  — Credit Risk Prediction Dashboard
# Run: streamlit run app/main.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from data.load_data import load_processed_data
from app.features import engineer_features, get_feature_names

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load models ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    models = {}
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith(".pkl") and fname != "feature_cols.pkl":
            name = fname.replace(".pkl", "")
            try:
                models[name] = joblib.load(os.path.join(MODELS_DIR, fname))
            except Exception:
                pass
    return models


@st.cache_resource
def load_feature_cols():
    path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_metrics():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(path):
        return pd.read_json(path)
    return pd.DataFrame()


@st.cache_data
def load_feature_importance():
    path = os.path.join(MODELS_DIR, "feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_data_for_eda():
    df = load_processed_data()
    return engineer_features(df)


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_risk(model, input_df: pd.DataFrame, feature_cols: list) -> dict:
    """Run prediction and return risk score + label."""
    # Align columns
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    prob     = model.predict_proba(input_df)[0][1]
    pred     = model.predict(input_df)[0]
    risk_pct = round(prob * 100, 1)

    if risk_pct < 30:
        risk_label = "🟢 Low Risk"
        color      = "green"
    elif risk_pct < 60:
        risk_label = "🟡 Medium Risk"
        color      = "orange"
    else:
        risk_label = "🔴 High Risk"
        color      = "red"

    return {
        "probability":  prob,
        "risk_pct":     risk_pct,
        "risk_label":   risk_label,
        "color":        color,
        "prediction":   pred,
    }


def build_input_features(
    duration, credit_amount, age, installment_rate,
    existing_credits, dependents, residence_since,
    checking_account, savings_account, employment_since,
    purpose, housing
) -> pd.DataFrame:
    """Build feature dict from sidebar inputs."""

    # Map UI labels to UCI codes
    checking_map = {
        "No account":      "A14",
        "< 0 DM":          "A11",
        "0-200 DM":        "A12",
        "> 200 DM":        "A13",
    }
    savings_map = {
        "Unknown/None":    "A61",
        "< 100 DM":        "A62",
        "100-500 DM":      "A63",
        "500-1000 DM":     "A64",
        "> 1000 DM":       "A65",
    }
    employment_map = {
        "Unemployed":      "A71",
        "< 1 year":        "A72",
        "1-4 years":       "A73",
        "4-7 years":       "A74",
        "> 7 years":       "A75",
    }
    purpose_map = {
        "Car (new)":       "A40",
        "Car (used)":      "A41",
        "Furniture":       "A42",
        "Electronics":     "A43",
        "Domestic":        "A44",
        "Education":       "A45",
    }
    housing_map = {
        "Rent":            "A151",
        "Own":             "A152",
        "Free":            "A153",
    }

    row = {
        "duration":           duration,
        "credit_amount":      credit_amount,
        "installment_rate":   installment_rate,
        "residence_since":    residence_since,
        "age":                age,
        "existing_credits":   existing_credits,
        "dependents":         dependents,
        "checking_account":   checking_map.get(checking_account, "A14"),
        "credit_history":     "A32",
        "purpose":            purpose_map.get(purpose, "A43"),
        "savings_account":    savings_map.get(savings_account, "A61"),
        "employment_since":   employment_map.get(employment_since, "A73"),
        "personal_status":    "A93",
        "other_debtors":      "A101",
        "property":           "A121",
        "other_installments": "A143",
        "housing":            housing_map.get(housing, "A152"),
        "job":                "A173",
        "telephone":          "A192",
        "foreign_worker":     "A201",
        "target":             0,
    }

    df_row = pd.DataFrame([row])

    # Apply same preprocessing as training
    cat_cols = [
        "checking_account", "credit_history", "purpose", "savings_account",
        "employment_since", "personal_status", "other_debtors", "property",
        "other_installments", "housing", "job", "telephone", "foreign_worker"
    ]
    num_cols = [
        "duration", "credit_amount", "installment_rate", "residence_since",
        "age", "existing_credits", "dependents"
    ]

    df_encoded = pd.get_dummies(df_row[cat_cols], drop_first=True)
    df_final   = pd.concat([df_row[num_cols], df_encoded], axis=1)
    df_final   = engineer_features(df_final.assign(target=0)).drop(columns=["target"], errors="ignore")

    return df_final


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    st.title("💳 Credit Risk Prediction Dashboard")
    st.caption("Ensemble ML models for credit default prediction — built for PayU-style lending use cases")

    models       = load_models()
    feature_cols = load_feature_cols()
    metrics_df   = load_metrics()

    if not models:
        st.error("⚠️ No trained models found. Run: `python app/train.py` first")
        st.code("python app/train.py")
        st.stop()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Predict Risk", "📊 Model Performance",
        "🔍 Feature Importance", "📈 Data Explorer", "🧠 SHAP Explainability"
    ])

    # ── Tab 1: Predict ────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Applicant Risk Scorer")
        st.caption("Enter loan applicant details to get a default risk score")

        col_form, col_result = st.columns([1, 1])

        with col_form:
            with st.form("prediction_form"):
                st.markdown("**Loan Details**")
                credit_amount    = st.slider("Loan Amount (₹)", 500, 20000, 5000, step=500)
                duration         = st.slider("Loan Duration (months)", 4, 72, 24)
                installment_rate = st.selectbox("Installment Rate (% of income)", [1, 2, 3, 4])
                purpose          = st.selectbox("Purpose", ["Car (new)", "Car (used)", "Furniture", "Electronics", "Domestic", "Education"])

                st.markdown("**Applicant Details**")
                age              = st.slider("Age", 18, 75, 35)
                checking_account = st.selectbox("Checking Account Balance", ["No account", "< 0 DM", "0-200 DM", "> 200 DM"])
                savings_account  = st.selectbox("Savings Account", ["Unknown/None", "< 100 DM", "100-500 DM", "500-1000 DM", "> 1000 DM"])
                employment_since = st.selectbox("Employment Duration", ["Unemployed", "< 1 year", "1-4 years", "4-7 years", "> 7 years"])
                housing          = st.selectbox("Housing", ["Rent", "Own", "Free"])
                existing_credits = st.number_input("Existing Credits at Bank", 1, 4, 1)
                dependents       = st.number_input("Number of Dependents", 1, 2, 1)
                residence_since  = st.number_input("Years at Current Residence", 1, 4, 2)

                model_choice = st.selectbox("Model", list(models.keys()))
                submitted    = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

        with col_result:
            if submitted:
                input_df = build_input_features(
                    duration, credit_amount, age, installment_rate,
                    existing_credits, dependents, residence_since,
                    checking_account, savings_account, employment_since,
                    purpose, housing,
                )

                result = predict_risk(models[model_choice], input_df, feature_cols)

                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result["risk_pct"],
                    title={"text": "Default Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": result["color"]},
                        "steps": [
                            {"range": [0, 30],  "color": "#d4edda"},
                            {"range": [30, 60], "color": "#fff3cd"},
                            {"range": [60, 100],"color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": result["risk_pct"],
                        }
                    },
                    number={"suffix": "%"},
                ))
                fig.update_layout(height=300, margin=dict(t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"### {result['risk_label']}")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Risk Score",    f"{result['risk_pct']}%")
                col_b.metric("Loan Amount",   f"₹{credit_amount:,}")
                col_c.metric("Monthly EMI",   f"₹{credit_amount // duration:,}")

                # Decision
                if result["risk_pct"] < 30:
                    st.success("✅ **Recommended: APPROVE** — Low default probability")
                elif result["risk_pct"] < 60:
                    st.warning("⚠️ **Recommended: REVIEW** — Moderate risk, consider additional verification")
                else:
                    st.error("❌ **Recommended: DECLINE** — High default probability")

                # Key risk factors
                st.markdown("**Key Risk Factors:**")
                factors = []
                if duration > 36:   factors.append("🔴 Long loan duration (>36 months)")
                if credit_amount > 10000: factors.append("🔴 High loan amount (>₹10,000)")
                if checking_account in ["No account", "< 0 DM"]: factors.append("🔴 Poor checking account status")
                if savings_account in ["Unknown/None", "< 100 DM"]: factors.append("🟡 Low savings")
                if age < 25:        factors.append("🟡 Young borrower (<25 years)")
                if existing_credits > 2: factors.append("🟡 Multiple existing credits")

                if factors:
                    for f in factors:
                        st.write(f)
                else:
                    st.write("✅ No major risk factors identified")
            else:
                st.info("👈 Fill in applicant details and click **Predict Risk**")

    # ── Tab 2: Model Performance ──────────────────────────────────────────────
    with tab2:
        st.subheader("Model Performance Comparison")

        if not metrics_df.empty:
            # Summary table
            display_cols = ["model", "auc_roc", "gini", "ks_stat", "precision", "recall"]
            available    = [c for c in display_cols if c in metrics_df.columns]
            st.dataframe(
                metrics_df[available].sort_values("auc_roc", ascending=False)
                .style.highlight_max(subset=["auc_roc", "gini", "ks_stat"], color="#d4edda")
                .format({c: "{:.4f}" for c in ["auc_roc", "gini", "ks_stat", "precision", "recall"] if c in available}),
                use_container_width=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                # AUC bar chart
                fig = px.bar(
                    metrics_df.sort_values("auc_roc"),
                    x="auc_roc", y="model", orientation="h",
                    title="AUC-ROC by Model",
                    color="auc_roc", color_continuous_scale="Greens",
                    labels={"auc_roc": "AUC-ROC", "model": "Model"},
                )
                fig.add_vline(x=0.75, line_dash="dash", annotation_text="Good (0.75)")
                fig.add_vline(x=0.80, line_dash="dot",  annotation_text="Strong (0.80)")
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # KS Statistic
                fig2 = px.bar(
                    metrics_df.sort_values("ks_stat"),
                    x="ks_stat", y="model", orientation="h",
                    title="KS Statistic by Model",
                    color="ks_stat", color_continuous_scale="Blues",
                    labels={"ks_stat": "KS Statistic", "model": "Model"},
                )
                fig2.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            # Metrics explanation
            with st.expander("📖 What do these metrics mean?"):
                st.markdown("""
**AUC-ROC** — Area Under the ROC Curve. Measures how well the model separates defaulters from non-defaulters.
- > 0.75 = Good | > 0.80 = Strong | > 0.85 = Excellent

**KS Statistic** — Kolmogorov-Smirnov. Industry standard in credit scoring.
Measures the maximum separation between default and non-default score distributions.
- > 0.30 = Good | > 0.40 = Strong

**Gini Coefficient** — `2 × AUC − 1`. Common in credit risk reporting.
- > 0.50 = Good | > 0.60 = Strong

**Precision** — Of predicted defaults, how many were actually defaults?

**Recall** — Of actual defaults, how many did we catch?
""")
        else:
            st.info("No metrics found. Run `python app/train.py` first.")

    # ── Tab 3: Feature Importance ─────────────────────────────────────────────
    with tab3:
        st.subheader("Feature Importance")
        imp_df = load_feature_importance()

        if not imp_df.empty:
            top_n = st.slider("Show top N features", 5, 30, 15)
            top   = imp_df.head(top_n)

            fig = px.bar(
                top, x="importance", y="feature", orientation="h",
                title=f"Top {top_n} Most Important Features",
                color="importance", color_continuous_scale="Viridis",
            )
            fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Higher importance = stronger influence on default prediction")
        else:
            st.info("Feature importance not found. Run `python app/train.py` first.")

    # ── Tab 4: Data Explorer ──────────────────────────────────────────────────
    with tab4:
        st.subheader("Dataset Explorer")

        try:
            df = load_data_for_eda()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Applicants",  len(df))
            col2.metric("Default Rate",      f"{df['target'].mean():.1%}")
            col3.metric("Total Features",    df.shape[1] - 1)
            col4.metric("Good Credit",       f"{(df['target']==0).sum()}")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    df, x="age", color=df["target"].map({0: "Good Credit", 1: "Default"}),
                    title="Age Distribution by Credit Outcome",
                    barmode="overlay", opacity=0.7,
                    color_discrete_map={"Good Credit": "#28a745", "Default": "#dc3545"},
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.histogram(
                    df, x="credit_amount",
                    color=df["target"].map({0: "Good Credit", 1: "Default"}),
                    title="Loan Amount Distribution by Credit Outcome",
                    barmode="overlay", opacity=0.7,
                    color_discrete_map={"Good Credit": "#28a745", "Default": "#dc3545"},
                )
                st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                fig3 = px.box(
                    df, x=df["target"].map({0: "Good Credit", 1: "Default"}),
                    y="duration", title="Loan Duration vs Credit Outcome",
                    color=df["target"].map({0: "Good Credit", 1: "Default"}),
                    color_discrete_map={"Good Credit": "#28a745", "Default": "#dc3545"},
                )
                st.plotly_chart(fig3, use_container_width=True)

            with col4:
                fig4 = px.box(
                    df, x=df["target"].map({0: "Good Credit", 1: "Default"}),
                    y="monthly_repayment", title="Monthly Repayment vs Credit Outcome",
                    color=df["target"].map({0: "Good Credit", 1: "Default"}),
                    color_discrete_map={"Good Credit": "#28a745", "Default": "#dc3545"},
                )
                st.plotly_chart(fig4, use_container_width=True)

        except Exception as e:
            st.error(f"Could not load data: {e}")


    # ── Tab 5: SHAP ───────────────────────────────────────────────────────────
    with tab5:
        st.subheader("🧠 SHAP Explainability")
        st.caption("Why did the model make this prediction?")
        try:
            import shap
            import matplotlib.pyplot as plt
            df = load_data_for_eda()
            X  = df.drop(columns=["target"])
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_cols]
            model_for_shap = models.get("random_forest") or models.get("gradient_boosting")
            if model_for_shap:
                with st.spinner("Computing SHAP values..."):
                    explainer   = shap.TreeExplainer(model_for_shap)
                    shap_obj    = explainer(X.iloc[:100])
                    shap_values = shap_obj.values
                    if shap_values.ndim == 3:
                        shap_values = shap_values[:, :, 1]
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X.iloc[:100], feature_names=list(X.columns), show=False, max_display=15)
                st.pyplot(fig)
                plt.close()
                st.caption("Red = increases default risk | Blue = decreases default risk")
        except Exception as e:
            st.error(f"SHAP error: {e}")


if __name__ == "__main__":
    main()
