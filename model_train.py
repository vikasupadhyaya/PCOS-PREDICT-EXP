import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# -------------------------------
# Directly define paths & config
# -------------------------------
DATA_PATH = "data/processed-data/PCOS_data_processed.csv"  # Path to processed CSV
MODEL_PATH = "models/pcos_random_forest.pkl"               # Where to save the model

TARGET_COLUMN = "PCOS (Y/N)"  # Target column in CSV

TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "models": {
        "logistic_regression": {
            "max_iter": 1000,
            "class_weight": "balanced"
        },
        "random_forest": {
            "n_estimators": 200,
            "random_state": 42,
            "class_weight": "balanced"
        },
        "xgboost": {
            "eval_metric": "logloss",
            "use_label_encoder": False
        }
    }
}

# -------------------------------
# Load processed data
# -------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Data loaded successfully. Shape: {df.shape}")

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TRAINING_CONFIG["test_size"],
    random_state=TRAINING_CONFIG["random_state"],
    stratify=y
)

# -------------------------------
# Logistic Regression
# -------------------------------
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(**TRAINING_CONFIG["models"]["logistic_regression"]))
])
log_reg.fit(X_train, y_train)
print("Logistic Regression Results:")
print(classification_report(y_test, log_reg.predict(X_test)))

# -------------------------------
# Random Forest
# -------------------------------
rf = RandomForestClassifier(**TRAINING_CONFIG["models"]["random_forest"])
rf.fit(X_train, y_train)
print("Random Forest Results:")
print(classification_report(y_test, rf.predict(X_test)))

# -------------------------------
# XGBoost
# -------------------------------
xgb_clf = xgb.XGBClassifier(**TRAINING_CONFIG["models"]["xgboost"])
xgb_clf.fit(X_train, y_train)
print("XGBoost Results:")
print(classification_report(y_test, xgb_clf.predict(X_test)))

# -------------------------------
# Save best model (Random Forest)
# -------------------------------
joblib.dump(rf, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
