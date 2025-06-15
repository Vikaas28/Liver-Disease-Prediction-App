import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

st.set_page_config(page_title="Liver Disease Prediction AI Assistant", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_liver_assets():
    model_path = os.path.join('liver_disease_model_assets', 'liver_prediction_xgb_model.pkl')
    scaler_path = os.path.join('liver_disease_model_assets', 'liver_scaler.pkl')
    imputer_path = os.path.join('liver_disease_model_assets', 'liver_imputer_ag_ratio.pkl')
    label_encoder_path = os.path.join('liver_disease_model_assets', 'liver_label_encoder_gender.pkl')
    gender_mapping_path = os.path.join('liver_disease_model_assets', 'liver_gender_mapping.pkl')
    metrics_path = os.path.join('liver_disease_model_assets', 'liver_model_evaluation_metrics.pkl')

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
        label_encoder = joblib.load(label_encoder_path)
        gender_mapping = joblib.load(gender_mapping_path)
        metrics = joblib.load(metrics_path)
        return model, scaler, imputer, label_encoder, gender_mapping, metrics
    except FileNotFoundError as e:
        st.error(f"Error: Missing model asset file! Please ensure the '{os.path.basename(e.filename)}' file is in the 'liver_disease_model_assets' directory.")
        st.info("Make sure you've run the model training script to save all necessary `.pkl` files.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading liver model assets: {e}")
        st.stop()

liver_model, liver_scaler, liver_imputer, liver_label_encoder, liver_gender_mapping, liver_eval_metrics = load_liver_assets()


st.title("🩺 Liver Disease Prediction AI Assistant")
st.markdown("---")

st.sidebar.header("Navigation")
page_selection = st.sidebar.radio("Go to", ["Dashboard & Model Performance", "Make a New Prediction"])

def preprocess_liver_input(input_data_df, scaler, imputer, label_encoder, feature_cols, gender_col='Gender'):
    processed_df = input_data_df.copy()

    if 'Albumin_and_Globulin_Ratio' in processed_df.columns:
        processed_df['Albumin_and_Globulin_Ratio'] = imputer.transform(processed_df[['Albumin_and_Globulin_Ratio']])

    if gender_col in processed_df.columns:
        try:
            processed_df[gender_col] = label_encoder.transform(processed_df[gender_col])
        except ValueError as e:
            st.error(f"Error in Gender encoding: {e}. Please ensure gender is 'Male' or 'Female'.")
            st.stop()

    try:
        processed_df = processed_df[feature_cols]
    except KeyError as e:
        st.error(f"Missing expected feature: {e}. Ensure all input fields are correctly named and present.")
        st.stop()

    scaled_data = scaler.transform(processed_df)
    return scaled_data

if page_selection == "Dashboard & Model Performance":
    st.header("Liver Disease Model Performance Overview")
    st.markdown("This section displays the performance metrics and visualizations of the trained model.")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{liver_eval_metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{liver_eval_metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{liver_eval_metrics['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{liver_eval_metrics['f1_score']:.4f}")
    with col5:
        st.metric("AUC-ROC", f"{liver_eval_metrics['roc_auc']:.4f}")

    st.markdown("---")

    col_charts_1, col_charts_2 = st.columns(2)

    with col_charts_1:
        st.subheader("Confusion Matrix")
        conf_matrix_np = np.array(liver_eval_metrics['conf_matrix'])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(conf_matrix_np, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted No Disease', 'Predicted Disease'],
                    yticklabels=['Actual No Disease', 'Actual Disease'], ax=ax_cm)
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)

    with col_charts_2:
        st.subheader("Receiver Operating Characteristic (ROC) Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        fpr_np = np.array(liver_eval_metrics['fpr'])
        tpr_np = np.array(liver_eval_metrics['tpr'])
        ax_roc.plot(fpr_np, tpr_np, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {liver_eval_metrics["roc_auc"]:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.markdown("---")

    if hasattr(liver_model, 'feature_importances_'):
        st.subheader("Feature Importance")
        feature_cols_model = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
            'Albumin_and_Globulin_Ratio'
        ]
        importances = liver_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_cols_model, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax_fi)
        ax_fi.set_title('Feature Importance (XGBoost)')
        ax_fi.set_xlabel('Relative Importance')
        ax_fi.set_ylabel('Feature')
        st.pyplot(fig_fi)
    else:
        st.info("Feature importance plot is not available for the selected model type or has not been implemented.")

elif page_selection == "Make a New Prediction":
    st.header("Predict Liver Disease Status")
    st.markdown("Enter patient's medical details below to get a liver disease prediction.")

    st.subheader("Patient Medical Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=100, value=45, step=1)
        gender = st.selectbox("Gender", options=list(liver_gender_mapping.keys()))
        total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, max_value=30.0, value=1.0, step=0.1, format="%.1f")
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=15.0, value=0.4, step=0.1, format="%.1f")

    with col2:
        alk_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, max_value=2000, value=180, step=1)
        alamine_amino = st.number_input("Alamine Aminotransferase", min_value=0, max_value=1000, value=20, step=1)
        aspartate_amino = st.number_input("Aspartate Aminotransferase", min_value=0, max_value=1000, value=25, step=1)
        total_proteins = st.number_input("Total Proteins (g/dL)", min_value=0.0, max_value=10.0, value=7.0, step=0.1, format="%.1f")


    with col3:
        albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=6.0, value=3.5, step=0.1, format="%.1f")
        albumin_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, max_value=2.0, value=0.9, step=0.01, format="%.2f")

    st.markdown("---")

    if st.button("Predict Liver Disease"):
        feature_cols_order = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin', # Correct spelling here
            'Albumin_and_Globulin_Ratio'
        ]

        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Total_Bilirubin': [total_bilirubin],
            'Direct_Bilirubin': [direct_bilirubin],
            'Alkaline_Phosphotase': [alk_phosphotase],
            'Alamine_Aminotransferase': [alamine_amino],
            'Aspartate_Aminotransferase': [aspartate_amino],
            'Total_Proteins': [total_proteins], # CORRECTED TYPO: 'Total_Protiens' -> 'Total_Proteins'
            'Albumin': [albumin],
            'Albumin_and_Globulin_Ratio': [albumin_globulin_ratio]
        })

        processed_input = preprocess_liver_input(
            input_data,
            liver_scaler,
            liver_imputer,
            liver_label_encoder,
            feature_cols_order
        )

        prediction = liver_model.predict(processed_input)[0]
        prediction_proba = liver_model.predict_proba(processed_input)[0, 1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"**Prediction: Patient likely has Liver Disease** (Probability: {prediction_proba:.2f})")
            st.write("Based on the provided data, the model indicates a high likelihood of liver disease. Further medical consultation and diagnostic tests are strongly recommended.")
        else:
            st.success(f"**Prediction: Patient likely does NOT have Liver Disease** (Probability: {prediction_proba:.2f})")
            st.write("Based on the provided data, the model indicates a low likelihood of liver disease. Continue to monitor health as advised by your doctor.")

        st.info("Disclaimer: This prediction is generated by a machine learning model and should be used as a supplementary tool. Always rely on professional medical advice and clinical judgment for diagnosis and treatment.")