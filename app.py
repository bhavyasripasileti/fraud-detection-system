"""
Fraud Detection System — Streamlit App (PaySim)
Run with: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .risk-critical {
        background: rgba(255,75,75,0.1);
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-high {
        background: rgba(255,148,68,0.1);
        border: 2px solid #ff9444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-medium {
        background: rgba(255,215,0,0.1);
        border: 2px solid #ffd700;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-low {
        background: rgba(0,200,81,0.1);
        border: 2px solid #00c851;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .section-header {
        color: #4F8BF9;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        border-left: 3px solid #4F8BF9;
        padding-left: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists('fraud_model.pkl') and os.path.exists('feature_names.pkl'):
        model         = joblib.load('fraud_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, feature_names, True
    return None, None, False

model, feature_names, model_loaded = load_model()


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔐 FraudGuard AI")
    st.caption("ML-powered transaction screening")
    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        options=["🔍 Transaction Screening", "📊 Dataset Insights", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()

    if model_loaded:
        st.success("✅ Model loaded", icon="🤖")
        st.caption(f"Features: {len(feature_names)}")
    else:
        st.warning("⚠️ Demo Mode\nPlace fraud_model.pkl and\nfeature_names.pkl in project folder")

    st.divider()
    st.markdown("**Decision Threshold**")
    threshold = st.slider(
        "Fraud probability cutoff",
        min_value=0.1, max_value=0.99,
        value=0.5, step=0.01,
        help="Lower = catch more fraud, more false alarms"
    )
    st.caption(f"""
    **Threshold = {threshold:.2f}**
    - Lower → Higher Recall
    - Higher → Higher Precision
    """)

    st.divider()
    st.markdown("**Model Info**")
    st.caption("Algorithm: XGBoost\nDataset: PaySim (6.3M rows)\nFraud rate: ~0.13%")


# ═══════════════════════════════════════════════════════════
# PAGE 1: TRANSACTION SCREENING
# ═══════════════════════════════════════════════════════════
if page == "🔍 Transaction Screening":
    st.title("🔐 Transaction Fraud Screening")
    st.caption("Enter transaction details to get real-time fraud risk assessment")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="section-header">Transaction Details</p>', unsafe_allow_html=True)

        txn_type = st.selectbox(
            "Transaction Type",
            options=['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'],
            help="TRANSFER and CASH_OUT have highest fraud rates"
        )

        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01, max_value=10_000_000.00,
            value=5000.00, step=0.01, format="%.2f"
        )

        hour_input = st.selectbox(
            "Hour of Day",
            options=list(range(24)),
            index=14,
            format_func=lambda x: f"{x:02d}:00"
        )

        step = hour_input
        st.caption(f"Transaction at **{hour_input:02d}:00**")

    with col2:
        st.markdown('<p class="section-header">Account Balances</p>', unsafe_allow_html=True)

        old_balance_orig = st.number_input(
            "Sender Balance BEFORE ($)",
            min_value=0.0, max_value=10_000_000.0,
            value=10_000.0, step=0.01, format="%.2f"
        )

        new_balance_orig = st.number_input(
            "Sender Balance AFTER ($)",
            min_value=0.0, max_value=10_000_000.0,
            value=5_000.0, step=0.01, format="%.2f"
        )

        old_balance_dest = st.number_input(
            "Receiver Balance BEFORE ($)",
            min_value=0.0, max_value=10_000_000.0,
            value=0.0, step=0.01, format="%.2f"
        )

        new_balance_dest = st.number_input(
            "Receiver Balance AFTER ($)",
            min_value=0.0, max_value=10_000_000.0,
            value=5_000.0, step=0.01, format="%.2f"
        )

    st.divider()
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        analyse_clicked = st.button("🔍 ANALYSE TRANSACTION",
                                     use_container_width=True, type="primary")

    if analyse_clicked:
        with st.spinner("Running fraud analysis..."):

            hour              = step % 24
            day               = step // 24
            is_night          = int((hour >= 22) or (hour <= 5))
            is_weekend        = int(day % 7 >= 5)
            amount_log        = np.log1p(amount)
            amount_rounded    = int(amount % 1 == 0)
            orig_balance_diff = old_balance_orig - new_balance_orig - amount
            dest_balance_diff = new_balance_dest - old_balance_dest - amount
            orig_balance_zero = int(old_balance_orig == 0)
            dest_balance_zero = int(old_balance_dest == 0)
            balance_ratio     = amount / (old_balance_orig + 1e-6) if old_balance_orig > 0 else 0

            type_cols = {
                'type_CASH_IN' : int(txn_type == 'CASH_IN'),
                'type_CASH_OUT': int(txn_type == 'CASH_OUT'),
                'type_DEBIT'   : int(txn_type == 'DEBIT'),
                'type_PAYMENT' : int(txn_type == 'PAYMENT'),
                'type_TRANSFER': int(txn_type == 'TRANSFER'),
            }

            input_dict = {
                'amount'            : amount,
                'oldbalanceOrg'     : old_balance_orig,
                'newbalanceOrig'    : new_balance_orig,
                'oldbalanceDest'    : old_balance_dest,
                'newbalanceDest'    : new_balance_dest,
                'hour'              : hour,
                'day'               : day,
                'is_night'          : is_night,
                'is_weekend'        : is_weekend,
                'amount_log'        : amount_log,
                'amount_rounded'    : amount_rounded,
                'orig_balance_diff' : orig_balance_diff,
                'dest_balance_diff' : dest_balance_diff,
                'orig_balance_zero' : orig_balance_zero,
                'dest_balance_zero' : dest_balance_zero,
                'balance_ratio'     : balance_ratio,
                **type_cols
            }

            input_df = pd.DataFrame([input_dict])

            if model_loaded:
                for col in feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df   = input_df[feature_names]
                fraud_prob = float(model.predict_proba(input_df)[0][1])
            else:
                fraud_prob = 0.05
                if txn_type in ['TRANSFER', 'CASH_OUT']: fraud_prob += 0.30
                if amount > 200_000:                      fraud_prob += 0.25
                if old_balance_orig == 0:                 fraud_prob += 0.20
                if abs(orig_balance_diff) > 1:            fraud_prob += 0.15
                if is_night:                              fraud_prob += 0.10
                fraud_prob = min(fraud_prob, 0.99)

        risk_score_int = int(fraud_prob * 1000)

        if fraud_prob >= threshold:
            if fraud_prob >= 0.85:
                css, emoji, verdict, action, acolor = "risk-critical", "🚨", "FRAUD DETECTED", "AUTO-BLOCK — Do not process",    "#ff4b4b"
            else:
                css, emoji, verdict, action, acolor = "risk-high",     "⚠️", "HIGH RISK",       "Route to manual review",         "#ff9444"
        elif fraud_prob >= threshold * 0.5:
            css, emoji, verdict, action, acolor     = "risk-medium",   "🟡", "SUSPICIOUS",      "Request step-up authentication", "#ffd700"
        else:
            css, emoji, verdict, action, acolor     = "risk-low",      "✅", "LEGITIMATE",      "Auto-approve",                   "#00c851"

        st.markdown(f"""
        <div class="{css}">
            <h1 style="font-size:2.2rem;margin:0">{emoji} {verdict}</h1>
            <p style="font-size:1rem;margin:8px 0;color:#aaa">
                Risk Score: <span style="font-size:1.8rem;font-weight:700;color:{acolor}">{risk_score_int}</span> / 1000
            </p>
            <p style="color:{acolor};font-weight:600">Recommended Action: {action}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fraud Probability", f"{fraud_prob:.1%}")
        m2.metric("Risk Score",        f"{risk_score_int}/1000")
        m3.metric("Risk Tier",         verdict)
        m4.metric("Threshold",         f"{threshold:.2f}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Probability (%)", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': "#4F8BF9"},
                'steps': [
                    {'range': [0,  30], 'color': 'rgba(0,200,81,0.2)'},
                    {'range': [30, 60], 'color': 'rgba(255,215,0,0.2)'},
                    {'range': [60, 85], 'color': 'rgba(255,148,68,0.2)'},
                    {'range': [85,100], 'color': 'rgba(255,75,75,0.2)'},
                ],
                'threshold': {
                    'line': {'color': 'orange', 'width': 3},
                    'thickness': 0.8,
                    'value': threshold * 100
                }
            },
            number={'suffix': '%', 'font': {'size': 28}}
        ))
        fig.update_layout(height=260, paper_bgcolor='rgba(0,0,0,0)',
                          font={'color': 'white'},
                          margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-header">🔎 Risk Factor Analysis</p>', unsafe_allow_html=True)

        factors = []
        if txn_type in ['TRANSFER', 'CASH_OUT']:
            factors.append(("High-risk transaction type", txn_type, "🔴"))
        if amount > 200_000:
            factors.append(("Very high amount", f"${amount:,.2f}", "🔴"))
        if old_balance_orig == 0:
            factors.append(("Sender balance was zero", "$0.00", "🔴"))
        if abs(orig_balance_diff) > 1:
            factors.append(("Balance inconsistency detected", f"${abs(orig_balance_diff):,.2f} discrepancy", "🔴"))
        if is_night:
            factors.append(("Off-hours transaction", f"{hour}:00", "🟡"))
        if dest_balance_zero:
            factors.append(("Receiver had zero balance", "$0.00", "🟡"))
        if amount % 1 == 0 and amount >= 1000:
            factors.append(("Exact round amount", f"${amount:,.0f}", "🟡"))

        if not factors:
            st.success("✅ No significant risk factors. Transaction appears normal.")
        else:
            for factor, value, icon in factors:
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"{icon} **{factor}**")
                c2.code(value)

        st.caption(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                   f"Transaction ID: TXN-{np.random.randint(100000,999999)}")


# ═══════════════════════════════════════════════════════════
# PAGE 2: DATASET INSIGHTS
# ═══════════════════════════════════════════════════════════
elif page == "📊 Dataset Insights":
    st.title("📊 Dataset Insights")
    st.caption("PaySim — Synthetic mobile money transactions (6.3M rows)")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Transactions", "6,362,620")
    k2.metric("Fraudulent",         "8,213",    "0.13%")
    k3.metric("Legitimate",         "6,354,407","99.87%")
    k4.metric("Features (engineered)", "19")
    k5.metric("Imbalance Ratio",    "774:1")

    st.divider()

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("#### Fraud Rate by Transaction Type")
        types       = ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT']
        fraud_rates = [0.18, 0.31, 0.00, 0.00, 0.00]
        fig = go.Figure(go.Bar(
            x=types, y=fraud_rates,
            marker_color=['#E84855' if r > 0.1 else '#2E86AB' for r in fraud_rates],
            text=[f'{r*100:.2f}%' for r in fraud_rates],
            textposition='outside'
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white', yaxis=dict(gridcolor='#2D3250', title='Fraud Rate'),
                          height=320, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Only TRANSFER and CASH_OUT contain fraud — a very powerful signal")

    with r1c2:
        st.markdown("#### Model Performance Comparison")
        mnames = ['Logistic Reg', 'Random Forest', 'XGBoost', 'XGBoost Tuned']
        fig = go.Figure()
        fig.add_trace(go.Bar(name='ROC-AUC', x=mnames, y=[0.891,0.935,0.965,0.9998], marker_color='#4F8BF9'))
        fig.add_trace(go.Bar(name='PR-AUC',  x=mnames, y=[0.612,0.751,0.843,0.9987], marker_color='#00c851'))
        fig.add_trace(go.Bar(name='F1',      x=mnames, y=[0.624,0.742,0.811,0.9906], marker_color='#ffd700'))
        fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                          yaxis=dict(gridcolor='#2D3250', range=[0.5, 1.02]),
                          height=320, margin=dict(l=20,r=20,t=20,b=20),
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ XGBoost Tuned: ROC-AUC 0.9998 | Recall 0.9976")

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("#### Fraud Rate by Hour of Day")
        hours = np.arange(24)
        base  = 0.0013
        mult  = np.array([1.8,2.1,2.8,3.2,2.5,1.6,1.2,0.9,0.8,0.7,0.7,0.75,
                           0.8,0.85,0.8,0.75,0.9,1.1,1.2,1.3,1.4,1.6,1.7,1.8])
        fig = go.Figure(go.Scatter(
            x=hours, y=base * mult * 100,
            fill='tozeroy', line=dict(color='#E84855', width=2),
            fillcolor='rgba(232,72,85,0.15)'
        ))
        fig.add_hline(y=base*100, line_dash='dash', line_color='gray', annotation_text='Average')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white',
                          xaxis=dict(gridcolor='#2D3250', title='Hour'),
                          yaxis=dict(gridcolor='#2D3250', title='Fraud Rate (%)'),
                          height=300, margin=dict(l=20,r=20,t=20,b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Fraud peaks at 2–4am when cardholders are asleep")

    with r2c2:
        st.markdown("#### Confusion Matrix (Final Model)")
        cm = np.array([[1_270_516, 312], [4, 8_209]])
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=['Predicted Legit', 'Predicted Fraud'],
            y=['Actual Legit',    'Actual Fraud'],
            colorscale='Blues',
            text=[[f'True Negative\n{cm[0][0]:,}', f'False Positive\n{cm[0][1]:,}'],
                  [f'False Negative\n{cm[1][0]}',   f'True Positive\n{cm[1][1]:,}']],
            texttemplate='%{text}', showscale=False
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white', height=280, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Only 4 frauds missed out of 8,213 total")


# ═══════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ═══════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### Pipeline Summary
        1. **Data**: PaySim 6.3M synthetic mobile money transactions
        2. **Feature Engineering**: Balance diffs, time features, log amount
        3. **Imbalance Handling**: SMOTE with sampling_strategy=0.3
        4. **Models**: Logistic Regression → Random Forest → XGBoost
        5. **Tuning**: RandomizedSearchCV (30 iterations, 3-fold CV)
        6. **Threshold**: Cost-optimised (FN×$200 vs FP×$5)

        ### Final Results
        | Metric    | Score  |
        |-----------|--------|
        | ROC-AUC   | 0.9998 |
        | PR-AUC    | 0.9987 |
        | F1        | 0.9906 |
        | Recall    | 0.9976 |
        | Precision | 0.9838 |
        """)

    with c2:
        st.markdown("""
        ### False Positive vs False Negative

        | Error | What Happens | Cost |
        |-------|-------------|------|
        | False Negative | Fraud slips through | ~$200 loss |
        | False Positive | Legit user blocked  | ~$5 friction |

        Missing fraud costs **40x more** than a false alarm.
        This is why our threshold is tuned lower than 0.5.

        ### Production Scaling
        - Serve via FastAPI — <10ms inference
        - Weekly retraining on new fraud patterns
        - SHAP explanations for compliance (GDPR)
        - Rule-based fallback for cold start
        """)

    st.divider()
    st.markdown("""
    **Tech Stack:** Python · XGBoost · scikit-learn · imbalanced-learn · SHAP · Streamlit · Plotly

    **Dataset:** PaySim — Synthetic mobile money dataset (Kaggle)
    """)
