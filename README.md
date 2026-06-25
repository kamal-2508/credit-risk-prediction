# Credit Risk Prediction — Ensemble ML Models

Predicts loan default probability using ensemble machine learning models.
Built to solve real fintech problems like those at PayU: *"Will this customer pay me back on time?"*

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

- **AUC-ROC** — model discrimination ability
- **KS Statistic** — industry standard in credit scoring
- **Gini Coefficient** — common in fintech reporting

## Quick Start

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train models (downloads data automatically)
python app/train.py

# 3. Run dashboard
streamlit run app/main.py

# 4. Run tests
pytest tests/ -v
```

## Project Structure

```
credit-risk-prediction/
├── app/
│   ├── train.py          # Training pipeline
│   ├── features.py       # Feature engineering
│   └── main.py           # Streamlit dashboard
├── data/
│   └── load_data.py      # Auto-downloads UCI dataset
├── models/               # Saved model files (auto-created)
├── tests/
│   └── test_credit_risk.py
└── requirements.txt
```

## Dataset

UCI German Credit Risk dataset — 1000 loan applicants, 20 features.
Downloads automatically, no Kaggle account needed.
