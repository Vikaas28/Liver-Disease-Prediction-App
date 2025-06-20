🩺 Liver Disease Prediction AI Assistant
This repository hosts a machine learning project designed to predict the likelihood of a patient having liver disease based on various medical parameters. The core of this project is an optimized XGBoost Classifier presented through an interactive Streamlit web application.

🎯 Project Goal
The primary goal of this project is to develop a reliable machine learning model that can assist healthcare professionals in the early screening and risk assessment of liver disease. Early detection is crucial for timely intervention and improved patient outcomes.

📊 Dataset
This project utilizes the Indian Liver Patient Dataset (ILPD), sourced from the UCI Machine Learning Repository (or similar public datasets). This dataset contains various features derived from blood tests and patient demographics, with the target variable indicating the presence or absence of liver disease.

Key Features include:

Age

Gender

Total_Bilirubin

Direct_Bilirubin

Alkaline_Phosphotase

Alamine_Aminotransferase

Aspartate_Aminotransferase

Total_Proteins

Albumin

Albumin_and_Globulin_Ratio

The target variable (Dataset) is binary: 1 for liver disease, 0 for no liver disease.

🧠 Methodology
The project follows a standard machine learning pipeline:

Data Preprocessing:

Missing Value Handling: Missing entries (specifically in Albumin_and_Globulin_Ratio) are imputed using the median strategy.

Categorical Encoding: The Gender feature is converted into a numerical representation using Label Encoding.

Outlier Detection: Outliers in numerical features are visualized using box plots. While detected, aggressive outlier removal or capping was not performed, as tree-based models like XGBoost are robust to them, and medical outliers can sometimes represent critical real-world cases.

Feature Scaling: All numerical features are scaled using StandardScaler to ensure they contribute equally to the model's learning process.

Model Selection & Training:

An XGBoost Classifier was chosen due to its high performance and robustness for classification tasks.

The data was split into training and testing sets (80% training, 20% testing) with stratification to maintain class balance.

Hyperparameter Tuning:

To achieve optimal accuracy, the XGBoost model's hyperparameters were fine-tuned using GridSearchCV with 5-fold cross-validation.

The optimization metric was AUC-ROC (scoring='roc_auc') to balance True Positive Rate and False Positive Rate, which is important for medical diagnosis.

Model Evaluation:

The performance of the best-tuned model was rigorously evaluated on the unseen test set using:

Accuracy

Precision

Recall

F1-Score

AUC-ROC (Area Under the Receiver Operating Characteristic Curve)

Confusion Matrix: To understand true positives, true negatives, false positives, and false negatives.

ROC Curve: A visual representation of the model's diagnostic ability.

🚀 Key Results
The model's performance metrics will be displayed directly on the Streamlit dashboard after it loads. The hyperparameter tuning process aims to find the best balance of these metrics for reliable prediction.

📂 Project Structure
.
├── ILPD.csv                      # The raw liver disease dataset
├── build_liver_model.py          # Script to load data, preprocess, train, tune, and save the model/assets
├── liver_app.py                  # The Streamlit web application dashboard
├── liver_disease_model_assets/   # Directory containing saved model, scaler, and other preprocessing objects (generated by build_liver_model.py)
│   ├── liver_prediction_xgb_model.pkl
│   ├── liver_scaler.pkl
│   ├── liver_imputer_ag_ratio.pkl
│   ├── liver_label_encoder_gender.pkl
│   ├── liver_gender_mapping.pkl
│   └── liver_model_evaluation_metrics.pkl
└── README.md                     # This file

Note: The .pkl files within liver_disease_model_assets/ are typically not committed to Git due to their size and the ability to regenerate them. They are generated when build_liver_model.py is run.

▶️ How to Run the Application
To run the interactive Streamlit dashboard:

Clone the Repository:

git clone https://github.com/YourUsername/Liver-Disease-Prediction-App.git
cd Liver-Disease-Prediction-App

(Replace YourUsername with your actual GitHub username.)

Install Dependencies:
It's highly recommended to use a virtual environment.

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib streamlit

Build/Re-train the Model and Generate Assets:
You need to run the training script first to create the necessary .pkl files in the liver_disease_model_assets folder.

python build_liver_model.py

This script will print messages as it saves the .pkl files to the liver_disease_model_assets directory.

Run the Streamlit App:

streamlit run liver_app.py

This command will open the dashboard in your default web browser.

🛠️ Technologies Used
Python 3.x

pandas: For data manipulation and analysis.

numpy: For numerical operations.

scikit-learn: For machine learning preprocessing (scaling, imputation, encoding), model selection (GridSearchCV), and evaluation metrics.

xgboost: The powerful gradient boosting library for model building.

matplotlib: For creating static, interactive, and animated visualizations.

seaborn: For drawing attractive statistical graphics.

joblib: For efficiently saving and loading Python objects (models, scalers).

streamlit: For building the interactive web application dashboard.

⚠️ Disclaimer
This AI assistant is for informational and educational purposes only and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns or before making any decisions related to your health or care. The predictions generated by this model are based on the provided data and machine learning algorithms and may not be 100% accurate.
