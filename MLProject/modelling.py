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

# --- KONFIGURASI DAGSHUB ---
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/treeasania/Model_Telco_Churn.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "treeasania"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2215b248b0302ccb69efecfff0d1d69bae7dceb0"

def train_skilled():
    print("=== Memulai Training Skilled (Manual Log + Tuning) ===")

    # 1. Load Data
    # Cek path dengan teliti (handle running dari root atau dari dalam folder)
    if os.path.exists("dataset_preprocessing/train_processed.csv"):
        train_path = "dataset_preprocessing/train_processed.csv"
        test_path = "dataset_preprocessing/test_processed.csv"
    elif os.path.exists("../dataset_preprocessing/train_processed.csv"):
        train_path = "../dataset_preprocessing/train_processed.csv"
        test_path = "../dataset_preprocessing/test_processed.csv"
    else:
        raise FileNotFoundError("File dataset tidak ditemukan! Cek struktur folder.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    # 2. Setup MLflow
    # PENTING: Jangan nyalakan set_experiment di sini biar 'nurut' sama GitHub Actions
    # mlflow.set_experiment("Eksperimen_Skilled_Telco")

    # Gunakan 'mlflow.start_run()' tanpa argumen aneh-aneh biar aman di CI
    with mlflow.start_run() as run:
        # 3. Hyperparameter Tuning
        print("Sedang melakukan GridSearch...")
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best Params ditemukan: {best_params}")

        # 4. Evaluasi & Hitung Metriks
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        # 5. Logging ke MLflow
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        
        # --- SAVE MODEL (BAGIAN PENTING YANG DIPERBAIKI) ---
        # Ganti "model" jadi "model_docker_local" supaya workflow GitHub menemukannya!
        print("Menyimpan model ke folder 'model_docker_local'...")
        mlflow.xgboost.log_model(best_model, "model_docker_local")
        print("Model berhasil disimpan.")

        print("Training Skilled Selesai. Metriks tercatat manual di MLflow.")

if __name__ == "__main__":
    print("Memulai Training via MLflow Project...")
    
    # Parse Argument (Biar MLflow Run tidak error saat passing argumen, walau tidak dipakai di GridSearch)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    # Jalankan Fungsi
    train_skilled()