import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix
import joblib
from datetime import datetime

# Configure paths
PROJECT_ROOT = Path.cwd()
FEATURE_DIR = PROJECT_ROOT / 'feature_matrices'
TFIDF_MAX_FEAT = 10000
TFIDF_NGRAMS = (1, 2)
SVD_COMPONENTS = 300
RANDOM_STATE = 42

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

# Initialize feature extractors
@st.cache_resource
def initialize_extractors():
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, ngram_range=TFIDF_NGRAMS, stop_words='english')
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    return tfidf, svd, scaler, ohe

tfidf, svd, scaler, ohe = initialize_extractors()

# Fit extractors on training data (approximate using saved X_all)
@st.cache_resource
def fit_extractors():
    try:
        X_all = np.load(FEATURE_DIR / 'X_all.npy')
        # Assume X_all structure: [SVD (300), categorical (~30), numerical (6)]
        num_cols = ['goal', 'pledged', 'usd_pledged', 'duration_days', 'pledged_to_goal', 'backers']
        cat_cols = ['main_category', 'currency', 'state']
        num_features = len(num_cols)
        cat_features = X_all.shape[1] - SVD_COMPONENTS - num_features
        # Dummy data to fit encoders (simplified)
        dummy_text = ['sample text'] * X_all.shape[0]
        dummy_num = X_all[:, -num_features:]  # Last 6 columns
        dummy_cat = pd.DataFrame({
            col: ['unknown'] * X_all.shape[0] for col in cat_cols
        })
        # Fit transformers
        X_tfidf = tfidf.fit_transform(dummy_text)
        svd.fit(X_tfidf)
        scaler.fit(dummy_num)
        ohe.fit(dummy_cat)
        return True
    except Exception as e:
        st.error(f"Error fitting extractors: {e}")
        return False

if not fit_extractors():
    st.stop()

# Streamlit app
st.title("Crowdfunding Fraud Detection")
st.write("Enter campaign details to predict fraud risk using Random Forest, XGBoost, and Logistic Regression models.")

# Input form
with st.form(key='campaign_form'):
    st.subheader("Campaign Details")
    blurb = st.text_area("Blurb or Name", placeholder="Enter campaign description or title")
    goal = st.number_input("Funding Goal ($)", min_value=0.0, value=1000.0)
    pledged = st.number_input("Amount Pledged ($)", min_value=0.0, value=0.0)
    backers = st.number_input("Number of Backers", min_value=0, value=0)
    main_category = st.selectbox("Main Category", [
        'Art', 'Comics', 'Crafts', 'Dance', 'Design', 'Fashion', 'Film & Video',
        'Food', 'Games', 'Journalism', 'Music', 'Photography', 'Publishing',
        'Technology', 'Theater', 'unknown'
    ])
    currency = st.selectbox("Currency", ['USD', 'GBP', 'EUR', 'CAD', 'AUD', 'unknown'])
    state = st.selectbox("Campaign State", ['failed', 'successful', 'canceled', 'live', 'suspended', 'unknown'])
    launched = st.date_input("Launched Date", value=datetime.now())
    deadline = st.date_input("Deadline Date", value=datetime.now())
    submit_button = st.form_submit_button(label='Predict Fraud')

# Process inputs and predict
if submit_button:
    try:
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

    except Exception as e:
        st.error(f"Error processing input: {e}")