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
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/processed-data/PCOS_data_processed.csv"
MODEL_PATH = "models/random_forest_model.pkl"

# -------------------------------
# Hyperparameters
# -------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

LOG_REG_PARAMS = {"C": 1.0, "max_iter": 1000}
RF_PARAMS = {"n_estimators": 100, "max_depth": None, "random_state": RANDOM_STATE}
XGB_PARAMS = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "use_label_encoder": False, "eval_metric": "logloss"}

TARGET_COLUMN = "Outcome"

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# -------------------------------
# MLflow Experiment
# -------------------------------
with mlflow.start_run():

    # -------------------------------
    # Logistic Regression
    # -------------------------------
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**LOG_REG_PARAMS))
    ])
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    print("Logistic Regression Results:")
    print(classification_report(y_test, y_pred_log))
    mlflow.log_metric("logreg_accuracy", accuracy_score(y_test, y_pred_log))

    # -------------------------------
    # Random Forest
    # -------------------------------
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Random Forest Results:")
    print(classification_report(y_test, y_pred_rf))
    mlflow.log_metric("rf_accuracy", accuracy_score(y_test, y_pred_rf))

    # -------------------------------
    # XGBoost
    # -------------------------------
    xgb_clf = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    print("XGBoost Results:")
    print(classification_report(y_test, y_pred_xgb))
    mlflow.log_metric("xgb_accuracy", accuracy_score(y_test, y_pred_xgb))

    # -------------------------------
    # Save the best model (Random Forest here)
    # -------------------------------
    joblib.dump(rf, MODEL_PATH)
    print(f"✅ Random Forest model saved to {MODEL_PATH}")
    mlflow.sklearn.log_model(rf, "random_forest_model")
