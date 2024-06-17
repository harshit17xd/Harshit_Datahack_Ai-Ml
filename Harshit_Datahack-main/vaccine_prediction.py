# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score

# Load data from CSV files
train_features = pd.read_csv('Dataset/training_set_features.csv')
train_labels = pd.read_csv('Dataset/training_set_labels.csv')
test_features = pd.read_csv('Dataset/test_set_features.csv')
submission_format = pd.read_csv('Dataset/submission_format.csv')

# Display the first few rows of the data
print("Training features preview:")
print(train_features.head())

print("Training labels preview:")
print(train_labels.head())

# Handle missing values by imputing with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
train_features_filled = imputer.fit_transform(train_features)
train_features_filled = pd.DataFrame(train_features_filled, columns=train_features.columns)

# Identify and encode categorical columns
categorical_cols = train_features_filled.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder().fit(train_features_filled[col].astype(str)) for col in categorical_cols}

for col, le in label_encoders.items():
    train_features_filled[col] = le.transform(train_features_filled[col].astype(str))

# Ensure there are no missing values after imputation
assert train_features_filled.isnull().sum().sum() == 0, "There are still missing values!"

# Define feature matrix X and target matrix y
X = train_features_filled
y = train_labels[['xyz_vaccine', 'seasonal_vaccine']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=7)

# Train the model with RandomForestClassifier wrapped in MultiOutputClassifier
rf = RandomForestClassifier(random_state=42)
multi_output_rf = MultiOutputClassifier(rf, n_jobs=-1)
multi_output_rf.fit(X_train, y_train)

# Predict probabilities on the validation set
y_val_pred = multi_output_rf.predict_proba(X_val)

# Extract probabilities for each target
y_val_pred_xyz = [pred[1] for pred in y_val_pred[0]]
y_val_pred_seasonal = [pred[1] for pred in y_val_pred[1]]

# Calculate ROC AUC scores
roc_auc_xyz = roc_auc_score(y_val['xyz_vaccine'], y_val_pred_xyz)
roc_auc_seasonal = roc_auc_score(y_val['seasonal_vaccine'], y_val_pred_seasonal)
mean_roc_auc = (roc_auc_xyz + roc_auc_seasonal) / 2

print(f"ROC AUC for xyz_vaccine: {roc_auc_xyz}")
print(f"ROC AUC for seasonal_vaccine: {roc_auc_seasonal}")
print(f"Mean ROC AUC: {mean_roc_auc}")

# Process the test set similarly to the training set
test_features_filled = imputer.transform(test_features)
test_features_filled = pd.DataFrame(test_features_filled, columns=test_features.columns)

for col, le in label_encoders.items():
    test_features_filled[col] = le.transform(test_features_filled[col].astype(str))

test_features_scaled = scaler.transform(test_features_filled)

# Predict probabilities on the test set
y_test_pred = multi_output_rf.predict_proba(test_features_scaled)

y_test_pred_xyz = [pred[1] for pred in y_test_pred[0]]
y_test_pred_seasonal = [pred[1] for pred in y_test_pred[1]]

# Create and save the submission file
submission = pd.DataFrame({
    'respondent_id': submission_format['respondent_id'],
    'xyz_vaccine': y_test_pred_xyz,
    'seasonal_vaccine': y_test_pred_seasonal
})

submission.to_csv('submission.csv', index=False)
print("Submission file created.")