# ─────────────────────────────────────────────────────────────────────────────
# api.py — FastAPI REST API for Credit Risk Prediction
#
# Run:   uvicorn api:app --reload
# Docs:  http://localhost:8000/docs  (auto-generated Swagger UI)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from app.features import engineer_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Ensemble ML models for credit default prediction — built for PayU-style lending use cases",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Load models on startup ───────────────────────────────────────────────────

models       = {}
feature_cols = []

@app.on_event("startup")
def load_models():
    global models, feature_cols

    fc_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    if not os.path.exists(fc_path):
        raise RuntimeError("Models not found. Run: python app/train.py first")

    feature_cols = joblib.load(fc_path)

    for fname in os.listdir(MODELS_DIR):
        if fname.endswith(".pkl") and fname != "feature_cols.pkl":
            name = fname.replace(".pkl", "")
            models[name] = joblib.load(os.path.join(MODELS_DIR, fname))

    print(f"✓ Loaded {len(models)} models: {list(models.keys())}")


# ─── Request / Response schemas ───────────────────────────────────────────────

class ApplicantInput(BaseModel):
    """Loan applicant details for risk prediction."""

    # Loan details
    credit_amount:    float = Field(..., gt=0,  description="Loan amount in INR", example=5000)
    duration:         int   = Field(..., gt=0,  description="Loan duration in months", example=24)
    installment_rate: int   = Field(..., ge=1, le=4, description="Installment rate % of income", example=2)
    purpose:          str   = Field(..., description="Loan purpose code", example="A43")

    # Applicant details
    age:              int   = Field(..., ge=18, le=100, description="Age in years", example=35)
    checking_account: str   = Field(..., description="Checking account status code", example="A14")
    savings_account:  str   = Field(..., description="Savings account status code", example="A61")
    employment_since: str   = Field(..., description="Employment duration code", example="A73")
    personal_status:  str   = Field("A93", description="Personal status code")
    other_debtors:    str   = Field("A101", description="Other debtors code")
    property:         str   = Field("A121", description="Property code")
    other_installments: str = Field("A143", description="Other installment plans code")
    housing:          str   = Field("A152", description="Housing code")
    job:              str   = Field("A173", description="Job code")
    telephone:        str   = Field("A192", description="Telephone code")
    foreign_worker:   str   = Field("A201", description="Foreign worker code")
    credit_history:   str   = Field("A32",  description="Credit history code")
    residence_since:  int   = Field(2, ge=1, le=4, description="Years at current residence")
    existing_credits: int   = Field(1, ge=1, le=4, description="Number of existing credits")
    dependents:       int   = Field(1, ge=1, le=2, description="Number of dependents")

    # Model selection
    model_name: str = Field("stacking_ensemble", description="Model to use for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "credit_amount":      5000,
                "duration":           24,
                "installment_rate":   2,
                "purpose":            "A43",
                "age":                35,
                "checking_account":   "A14",
                "savings_account":    "A61",
                "employment_since":   "A73",
                "housing":            "A152",
                "model_name":         "stacking_ensemble",
            }
        }


class PredictionResponse(BaseModel):
    """Credit risk prediction result."""
    applicant_id:     str
    model_used:       str
    risk_score:       float = Field(..., description="Default probability 0-100%")
    risk_label:       str   = Field(..., description="Low / Medium / High Risk")
    decision:         str   = Field(..., description="APPROVE / REVIEW / DECLINE")
    default_probability: float = Field(..., description="Raw probability 0-1")
    key_risk_factors: list[str]
    timestamp:        str


class ModelInfo(BaseModel):
    name:    str
    loaded:  bool


class MetricsResponse(BaseModel):
    model:     str
    auc_roc:   Optional[float]
    gini:      Optional[float]
    ks_stat:   Optional[float]
    precision: Optional[float]
    recall:    Optional[float]


# ─── Helper functions ─────────────────────────────────────────────────────────

def build_feature_df(inp: ApplicantInput) -> pd.DataFrame:
    """Convert API input to model-ready DataFrame."""

    row = {
        "duration":           inp.duration,
        "credit_amount":      inp.credit_amount,
        "installment_rate":   inp.installment_rate,
        "residence_since":    inp.residence_since,
        "age":                inp.age,
        "existing_credits":   inp.existing_credits,
        "dependents":         inp.dependents,
        "checking_account":   inp.checking_account,
        "credit_history":     inp.credit_history,
        "purpose":            inp.purpose,
        "savings_account":    inp.savings_account,
        "employment_since":   inp.employment_since,
        "personal_status":    inp.personal_status,
        "other_debtors":      inp.other_debtors,
        "property":           inp.property,
        "other_installments": inp.other_installments,
        "housing":            inp.housing,
        "job":                inp.job,
        "telephone":          inp.telephone,
        "foreign_worker":     inp.foreign_worker,
        "target":             0,
    }

    df = pd.DataFrame([row])

    # One-hot encode categoricals
    cat_cols = [
        "checking_account", "credit_history", "purpose", "savings_account",
        "employment_since", "personal_status", "other_debtors", "property",
        "other_installments", "housing", "job", "telephone", "foreign_worker"
    ]
    num_cols = [
        "duration", "credit_amount", "installment_rate", "residence_since",
        "age", "existing_credits", "dependents"
    ]

    df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)
    df_final   = pd.concat([df[num_cols], df_encoded], axis=1)
    df_final   = engineer_features(df_final.assign(target=0)).drop(columns=["target"], errors="ignore")

    # Align with training features
    for col in feature_cols:
        if col not in df_final.columns:
            df_final[col] = 0
    df_final = df_final[feature_cols]

    return df_final


def get_risk_factors(inp: ApplicantInput) -> list[str]:
    """Return human-readable risk factors."""
    factors = []
    if inp.duration > 36:
        factors.append("Long loan duration (>36 months)")
    if inp.credit_amount > 10000:
        factors.append("High loan amount (>10000)")
    if inp.checking_account in ["A11", "A14"]:
        factors.append("Poor or no checking account")
    if inp.savings_account in ["A61", "A62"]:
        factors.append("Low savings")
    if inp.age < 25:
        factors.append("Young borrower (<25 years)")
    if inp.existing_credits > 2:
        factors.append("Multiple existing credits")
    if inp.installment_rate >= 4:
        factors.append("High installment rate")
    return factors if factors else ["No major risk factors identified"]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message":  "Credit Risk Prediction API",
        "version":  "1.0.0",
        "docs":     "/docs",
        "models":   list(models.keys()),
    }


@app.get("/health", tags=["General"])
def health():
    return {
        "status":       "healthy",
        "models_loaded": len(models),
        "timestamp":    datetime.now().isoformat(),
    }


@app.get("/models", response_model=list[ModelInfo], tags=["Models"])
def list_models():
    """List all available trained models."""
    return [{"name": name, "loaded": True} for name in models.keys()]


@app.get("/metrics", response_model=list[MetricsResponse], tags=["Models"])
def get_metrics():
    """Get performance metrics for all models."""
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not found. Run training first.")

    df = pd.read_json(metrics_path)
    return df.to_dict(orient="records")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(inp: ApplicantInput):
    """
    Predict credit default risk for a loan applicant.

    Returns:
    - risk_score: 0-100% default probability
    - risk_label: Low / Medium / High
    - decision: APPROVE / REVIEW / DECLINE
    - key_risk_factors: list of risk drivers
    """
    # Validate model
    if inp.model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{inp.model_name}' not found. Available: {list(models.keys())}"
        )

    # Build features
    try:
        df = build_feature_df(inp)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {str(e)}")

    # Predict
    model = models[inp.model_name]
    prob  = float(model.predict_proba(df)[0][1])
    score = round(prob * 100, 2)

    # Risk label
    if score < 30:
        risk_label = "Low Risk"
        decision   = "APPROVE"
    elif score < 60:
        risk_label = "Medium Risk"
        decision   = "REVIEW"
    else:
        risk_label = "High Risk"
        decision   = "DECLINE"

    return PredictionResponse(
        applicant_id=        f"APP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        model_used=          inp.model_name,
        risk_score=          score,
        risk_label=          risk_label,
        decision=            decision,
        default_probability= round(prob, 4),
        key_risk_factors=    get_risk_factors(inp),
        timestamp=           datetime.now().isoformat(),
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(applicants: list[ApplicantInput]):
    """Predict risk for multiple applicants at once."""
    if len(applicants) > 100:
        raise HTTPException(status_code=400, detail="Max 100 applicants per batch request")

    results = []
    for inp in applicants:
        try:
            result = predict(inp)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "model_name": inp.model_name})

    return {"total": len(results), "results": results}
