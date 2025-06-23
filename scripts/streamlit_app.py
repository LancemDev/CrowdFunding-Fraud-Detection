import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
import joblib
from datetime import datetime

# Configure paths
PROJECT_ROOT = Path.cwd()
FEATURE_DIR = PROJECT_ROOT / 'feature_matrices'

# Load models
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load(FEATURE_DIR / 'rf_fraud_model.pkl')
        xgb_model = joblib.load(FEATURE_DIR / 'xgb_fraud_model.pkl')
        lr_model = joblib.load(FEATURE_DIR / 'lr_fraud_model.pkl')
        return rf_model, xgb_model, lr_model
    except FileNotFoundError as e:
        st.error(f"Model files not found in {FEATURE_DIR}: {e}")
        return None, None, None

rf_model, xgb_model, lr_model = load_models()
if all(model is None for model in [rf_model, xgb_model, lr_model]):
    st.stop()

# Load feature extractors
@st.cache_resource
def load_extractors():
    try:
        tfidf = joblib.load(FEATURE_DIR / 'tfidf.pkl')
        svd = joblib.load(FEATURE_DIR / 'svd.pkl')
        scaler = joblib.load(FEATURE_DIR / 'scaler.pkl')
        ohe = joblib.load(FEATURE_DIR / 'ohe.pkl')
        return tfidf, svd, scaler, ohe
    except FileNotFoundError as e:
        st.error(f"Extractor files not found in {FEATURE_DIR}: {e}. Please save tfidf.pkl, svd.pkl, scaler.pkl, ohe.pkl using the notebook.")
        return None, None, None, None

tfidf, svd, scaler, ohe = load_extractors()
if all(extractor is None for extractor in [tfidf, svd, scaler, ohe]):
    st.stop()

# Streamlit app
st.title("Crowdfunding Fraud Detection")
st.write("Enter campaign details to predict the likelihood of fraud using Random Forest, XGBoost, and Logistic Regression models.")
st.markdown("**Note**: XGBoost is the most reliable model (F1-Score ~0.90 for fraud detection).")

# Input form
with st.form(key='campaign_form'):
    st.subheader("Campaign Details")
    blurb = st.text_area("Campaign Description or Title", placeholder="e.g., Revolutionary smartwatch with advanced features")
    goal = st.number_input("Funding Goal ($)", min_value=0.0, value=1000.0, step=100.0)
    pledged = st.number_input("Amount Pledged ($)", min_value=0.0, value=0.0, step=10.0)
    backers = st.number_input("Number of Backers", min_value=0, value=0, step=1)
    main_category = st.selectbox("Main Category", [
        'Art', 'Comics', 'Crafts', 'Dance', 'Design', 'Fashion', 'Film & Video',
        'Food', 'Games', 'Journalism', 'Music', 'Photography', 'Publishing',
        'Technology', 'Theater', 'unknown'
    ])
    currency = st.selectbox("Currency", ['USD', 'GBP', 'EUR', 'CAD', 'AUD', 'unknown'])
    state = st.selectbox("Campaign State", ['failed', 'successful', 'canceled', 'live', 'suspended', 'unknown'])
    launched = st.date_input("Launched Date", value=datetime.now())
    deadline = st.date_input("Deadline Date", value=datetime.now())
    submit_button = st.form_submit_button(label="Predict Fraud Risk")

# Process inputs and predict
if submit_button:
    try:
        # Validate inputs
        if deadline < launched:
            st.error("Deadline must be after launched date.")
            st.stop()
        if goal < 0 or pledged < 0 or backers < 0:
            st.error("Goal, pledged, and backers must be non-negative.")
            st.stop()
        if not blurb.strip():
            st.warning("Description is empty. Using default empty text.")

        # Create input DataFrame
        input_data = pd.DataFrame({
            'blurb': [blurb if blurb else ''],
            'name': [blurb if blurb else ''],
            'goal': [float(goal)],
            'pledged': [float(pledged)],
            'backers': [float(backers)],
            'usd_pledged': [float(pledged)],  # Approximate
            'main_category': [main_category],
            'currency': [currency],
            'state': [state],
            'launched': [pd.to_datetime(launched)],
            'deadline': [pd.to_datetime(deadline)]
        })

        # Engineer features
        input_data['duration_days'] = (input_data['deadline'] - input_data['launched']).dt.days.clip(lower=1)
        input_data['pledged_to_goal'] = input_data['pledged'] / input_data['goal'].replace(0, 1)

        # Preprocess features
        text_col = 'blurb' if blurb.strip() else 'name'
        num_cols = ['goal', 'pledged', 'usd_pledged', 'duration_days', 'pledged_to_goal', 'backers']
        cat_cols = ['main_category', 'currency', 'state']

        # Text features
        X_tfidf = tfidf.transform(input_data[text_col])
        X_svd = svd.transform(X_tfidf)

        # Numerical features
        X_num = scaler.transform(input_data[num_cols])

        # Categorical features
        X_cat = ohe.transform(input_data[cat_cols])

        # Combine features
        X_input = hstack([csr_matrix(X_svd), X_cat, csr_matrix(X_num)]).tocsr()

        # Predict with all models
        models = {
            'Random Forest': rf_model,
            'XGBoost': xgb_model,
            'Logistic Regression': lr_model
        }
        st.subheader("Prediction Results")
        results = {}
        for name, model in models.items():
            try:
                prob = model.predict_proba(X_input)[0, 1] * 100
                risk = "High" if prob > 50 else "Low"
                results[name] = prob
                st.write(f"**{name}**: Fraud Probability = {prob:.2f}% ({risk} Risk)")
            except Exception as e:
                st.error(f"Error predicting with {name}: {e}")

        # Display comparison
        if results:
            results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Fraud Probability (%)'])
            st.write("**Model Comparison**")
            st.dataframe(results_df.style.format({'Fraud Probability (%)': '{:.2f}'}))

            # Bar chart
            st.write("**Fraud Probability Visualization**")
            st.bar_chart(results_df)

    except Exception as e:
        st.error(f"Error processing input: {e}")