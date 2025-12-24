# -*- coding: utf-8 -*-
import argparse
import sys
import os
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

# --- FUNGSI LOAD DATA (Bisa baca dari dalam/luar folder) ---
def load_data():
    # Cek path dataset dengan teliti
    path1 = "dataset_preprocessing/train_processed.csv"
    path2 = "../dataset_preprocessing/train_processed.csv"
    
    if os.path.exists(path1):
        return pd.read_csv(path1), pd.read_csv(path1.replace("train", "test"))
    elif os.path.exists(path2):
        return pd.read_csv(path2), pd.read_csv(path2.replace("train", "test"))
    else:
        raise FileNotFoundError("Dataset tidak ditemukan! Cek path folder.")

def train_skilled():
    print("=== Memulai Training Skilled (Fix CI/CD) ===")

    # 1. Load Data
    train_df, test_df = load_data()

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    # 2. Setup MLflow
    # PENTING: Jangan ada set_experiment. 
    # Gunakan start_run() kosong agar dia otomatis nempel ke GitHub Actions.
    
    with mlflow.start_run() as run:
        print(f"Run ID: {run.info.run_id}")
        
        # 3. Hyperparameter Tuning (GridSearch)
        print("Sedang melakukan GridSearch...")
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        
        # cv=2 biar cepat selesai di GitHub
        grid_search = GridSearchCV(clf, param_grid, cv=2, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best Params: {best_params}")

        # 4. Evaluasi
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        # 5. Logging
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        
        # --- SAVE MODEL ---
        # Nama folder HARUS 'model_docker_local' agar terbaca oleh main.yml
        print("Menyimpan model ke 'model_docker_local'...")
        mlflow.xgboost.log_model(best_model, "model_docker_local")
        print("✅ Model berhasil disimpan.")

if __name__ == "__main__":
    # Parser argumen dummy (supaya tidak error kalau dipanggil dengan argumen)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    train_skilled()