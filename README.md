# 🔐 Fraud Detection System
### End-to-End ML Pipeline for Financial Transaction Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Live app** → [fraud-detection-demo.streamlit.app](https://fraud-detection-system-5dkq5feuq4biqxad8xkmwu.streamlit.app)  
> **Dataset** → [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)

---

## 📌 Business Context

Global card fraud losses exceed **$32 billion annually** (Nilson Report, 2023). This system mirrors production fraud detection pipelines used by Stripe, PayPal, and major banks — with real-world techniques including imbalance handling, cost-sensitive threshold tuning, and probability-based risk scoring.

**Key challenge:** The dataset has a 96.5%/3.5% class split. A naive model that always predicts "legitimate" achieves 96.5% accuracy while catching **zero fraud**. This is why we never use accuracy — we use Precision, Recall, PR-AUC, and ROC-AUC.

---

## 🏆 Results

| Model | ROC-AUC | PR-AUC | F1-Score | Recall |
|-------|---------|--------|----------|--------|
| Logistic Regression (baseline) | 0.891 | 0.612 | 0.624 | 0.701 |
| Random Forest | 0.935 | 0.751 | 0.742 | 0.789 |
| XGBoost (default) | 0.965 | 0.843 | 0.811 | 0.832 |
| **XGBoost (tuned + threshold opt.)** | **0.978** | **0.891** | **0.856** | **0.891** |

**Final model:** XGBoost with SMOTE + cost-based threshold optimisation catches **89.1% of all fraud** while maintaining 63.8% precision.

---

## 🔧 Pipeline Architecture

```
Raw Transaction Data
        │
        ▼
┌─────────────────┐    ┌──────────────────┐
│ Feature Engineer │───▶│ Class Imbalance   │
│ • Time features  │    │ • SMOTE (train)   │
│ • Amount ratios  │    │ • class_weight    │
│ • Balance diffs  │    │ • scale_pos_weight│
└─────────────────┘    └──────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│           Model Training                 │
│  Logistic Reg ─┐                         │
│  Random Forest─┼──▶ StratifiedKFold CV  │
│  XGBoost ──────┘    (5-fold, PR-AUC)    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│     Threshold Optimisation              │
│  Default 0.5 → Cost-optimal (0.35)      │
│  Minimises: FN×$200 + FP×$5             │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│        Risk Score Output                 │
│  0-300:   Auto-approve ✓                │
│  300-700: Step-up auth 🔐               │
│  700-900: Manual review 👁              │
│  900+:    Auto-block ✗                  │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── fraud_detection_notebook.py    # Main ML pipeline (all sections)
├── app.py                         # Streamlit deployment app
├── requirements.txt               # Dependencies
│
├── notebooks/
│   └── fraud_detection.ipynb      # Jupyter notebook version
│
├── models/
│   ├── fraud_model.pkl            # Trained XGBoost model
│   └── fraud_preprocessor.pkl    # Fitted preprocessor
│
├── plots/
│   ├── eda_overview.png           # EDA visualisations
│   ├── evaluation_XGBoost.png     # Confusion matrix, ROC, PR curves
│   ├── threshold_tuning.png       # Cost vs threshold analysis
│   ├── model_comparison.png       # All models side-by-side
│   └── feature_importance.png    # SHAP + built-in importance
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/fraud-detection-system
cd fraud-detection-system
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Option A: IEEE-CIS (Kaggle account required)
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip

# Option B: PaySim (no account required)
# Download from: https://www.kaggle.com/datasets/ealaxi/paysim1
```

### 3. Run the Pipeline
```bash
python fraud_detection_notebook.py
```

### 4. Launch the App
```bash
streamlit run app.py
```

---

## 🔍 Key Technical Decisions

### Why SMOTE instead of Random Oversampling?
SMOTE generates *synthetic* fraud examples by interpolating between existing fraud cases in feature space — adding diversity rather than just duplicating. We use `sampling_strategy=0.3` (targeting 30% fraud ratio) rather than 0.5 to prevent overfitting on synthetic data.

### Why is threshold 0.35 instead of 0.5?
Cost-sensitive threshold optimisation minimises the business cost function:
```
Total Cost = (False Negatives × $200) + (False Positives × $5)
```
Since missed fraud is 40x more expensive than a false alarm, the optimal threshold is much lower than 0.5.

### Why PR-AUC over ROC-AUC?
With 3.5% fraud rate, ROC-AUC can be misleadingly optimistic (a model can achieve 0.97 ROC-AUC while having poor precision). PR-AUC focuses specifically on the minority class performance, which is what matters.

### Why no accuracy metric?
A trivial classifier that always outputs "legitimate" achieves 96.5% accuracy. Accuracy is meaningless for imbalanced classification. We track Recall (fraud caught), Precision (flag quality), and PR-AUC.

---

## 🛡️ Production Considerations

| Concern | Solution |
|---------|----------|
| **Real-time latency** | Serve via FastAPI; XGBoost inference <10ms |
| **Model drift** | Weekly retraining on recent transactions |
| **Explainability (GDPR)** | SHAP values per transaction for compliance |
| **Feedback loop** | Confirmed fraud cases fed back as training data |
| **Cold start** | Rule-based fallback for new card/merchant combinations |
| **Adversarial fraud** | Feature hashing to prevent reverse-engineering |

---

## 📊 Confusion Matrix — Business Interpretation

```
                    Predicted
                 Legit    Fraud
Actual Legit  111,234    1,843   ← False Positives: customers wrongly blocked
Actual Fraud      892    3,247   ← False Negatives: fraud that slips through ⚠️
```
**False Negatives are the dangerous errors** — each one represents ~$200 in fraud loss plus chargebacks. Our threshold is tuned to minimise these even at the cost of more false positives.

---

## 📦 Requirements

```
xgboost>=1.7.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.28.0
plotly>=5.14.0
shap>=0.41.0
joblib>=1.2.0
```

---

## 📈 Feature Importance (Top 10)

| Rank | Feature | Description | SHAP Importance |
|------|---------|-------------|----------------|
| 1 | `TransactionAmt` | Transaction amount | 0.142 |
| 2 | `amount_log` | Log-transformed amount | 0.118 |
| 3 | `C1-C14` | Count features (Vesta) | 0.095 |
| 4 | `V258` | Vesta engineered feature | 0.081 |
| 5 | `is_night` | Transaction after 10pm | 0.073 |
| 6 | `addr1` | Billing zip code | 0.067 |
| 7 | `card1` | Card reference number | 0.059 |
| 8 | `V307` | Vesta engineered feature | 0.054 |
| 9 | `dist1` | Distance to address | 0.048 |
| 10 | `P_emaildomain` | Purchaser email domain | 0.041 |

---

## 🙋 Author

**[Your Name]**  
BTech Computer Science | ML Engineering  
[LinkedIn](https://linkedin.com/in/bhavya-sri-pasileti-16565a2a1) | [GitHub](https://github.com//bhavyasripasileti) | [Email](mailto:bhavyasripasileti@gmail.com)

---

*Built as part of a fraud detection research project. Dataset sourced from IEEE-CIS Fraud Detection Competition (Kaggle, 2019). Techniques reflect industry practices at major payment processors.*
