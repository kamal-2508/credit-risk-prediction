# Credit Risk Prediction — Ensemble ML Models

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/kamalrajn/credit-risk-prediction)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/kamal-2508/credit-risk-prediction)

Predicts loan default probability using ensemble machine learning models.
Built to solve real fintech problems like those at PayU: *"Will this customer pay me back on time?"*

## 🚀 Live Demo
👉 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/kamalrajn/credit-risk-prediction)**

## What it does

| Input | Output |
|-------|--------|
| Loan applicant details (age, income, loan amount, duration) | Default probability score (0–100%) |
| | Risk label: Low / Medium / High |
| | Approval recommendation |

## Models

| Model | Type |
|-------|------|
| Logistic Regression | Baseline |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| XGBoost | Boosting |
| LightGBM | Boosting |
| **Stacking Ensemble** | **Meta-learner (best)** |

## Key Metrics

| Metric | Score | Benchmark |
|--------|-------|-----------|
| AUC-ROC | 0.7995 | > 0.75 = Good ✅ |
| KS Statistic | 0.4786 | > 0.40 = Strong ✅ |
| Gini Coefficient | 0.5990 | > 0.50 = Good ✅ |

## Features

- 🔮 **Risk Scorer** — live default probability with gauge chart
- 📊 **Model Performance** — AUC, KS, Gini comparison across all models
- 🔍 **Feature Importance** — top drivers of default risk
- 📈 **Data Explorer** — visualize dataset distributions
- 🧠 **SHAP Explainability** — why was this applicant approved/rejected?

## Quick Start

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train models (downloads data automatically)
python app/train.py

# 3. Run dashboard
streamlit run app.py

# 4. Run tests
pytest tests/ -v
```

## Project Structure

```
credit-risk-prediction/
├── app.py                    # Entry point (HF Spaces)
├── app/
│   ├── train.py              # Training pipeline (6 models)
│   ├── features.py           # Feature engineering
│   └── main.py               # Streamlit dashboard
├── data/
│   └── load_data.py          # Auto-downloads UCI dataset
├── models/                   # Saved model files (auto-created)
├── tests/
│   └── test_credit_risk.py   # 18 pytest cases
└── requirements.txt
```

## Dataset

UCI German Credit Risk dataset — 1000 loan applicants, 20 features.
Downloads automatically, no Kaggle account needed.
