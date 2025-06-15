print("\n--- BUILDING AND TRAINING THE XGBOOST MODEL ---")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
import xgboost as xgb


# Initialize XGBoost Classifier

liver_xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', # For binary classification
    eval_metric='logloss',       # Evaluation metric during training
    use_label_encoder=False,     # Suppress deprecation warning
    n_estimators=200,            # Number of boosting rounds
    learning_rate=0.1,           # Step size shrinkage to prevent overfitting
    random_state=42              # For reproducibility
)

# Train the model
liver_xgb_model.fit(X_train, y_train)

print("XGBoost model training complete!")
# Make predictions on the test set
y_pred_liver = liver_xgb_model.predict(X_test)
y_prob_liver = liver_xgb_model.predict_proba(X_test)[:, 1] # Probability of liver disease (class 1)

# Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred_liver)
precision = precision_score(y_test, y_pred_liver)
recall = recall_score(y_test, y_pred_liver)
f1 = f1_score(y_test, y_pred_liver)
roc_auc = roc_auc_score(y_test, y_prob_liver)
conf_matrix = confusion_matrix(y_test, y_pred_liver)
fpr, tpr, thresholds = roc_curve(y_test, y_prob_liver) # For ROC plot

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_liver))

print("\nConfusion Matrix:")
print(conf_matrix)
# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted No Disease', 'Predicted Disease'],
            yticklabels=['Actual No Disease', 'Actual Disease'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Liver Disease Prediction')
plt.show()
# Plot ROC Curve
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


print("\n--- PERFORMING HYPERPARAMETER TUNING (GridSearchCV) ---")

# Initialize XGBoost Classifier with default parameters (will be tuned)
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)

# Define the parameter grid for GridSearchCV
# Keeping the grid small for demonstration purposes. For production, you might explore wider ranges.
param_grid = {
    'n_estimators': [50, 100, 200], # Number of boosting rounds
    'learning_rate': [0.05, 0.1, 0.2], # Step size shrinkage
    'max_depth': [3, 5, 7],         # Maximum depth of a tree
    'subsample': [0.7, 0.9],        # Subsample ratio of the training instance
    'colsample_bytree': [0.7, 0.9]  # Subsample ratio of columns when constructing each tree
}

# Setup GridSearchCV
grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='roc_auc', # Optimize for AUC-ROC score
    cv=5,              # 5-fold cross-validation
    n_jobs=-1,         # Use all available CPU cores
    verbose=1          # Print progress messages
)

# Fit GridSearchCV to the training data
grid_search_xgb.fit(X_train, y_train)

# Get the best estimator and its parameters
best_xgb_model = grid_search_xgb.best_estimator_
best_params = grid_search_xgb.best_params_

print("\n--- Hyperparameter Tuning Complete ---")
print(f"Best AUC-ROC Score during CV: {grid_search_xgb.best_score_:.4f}")
print(f"Best Parameters Found: {best_params}")
