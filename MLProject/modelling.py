# -*- coding: utf-8 -*-
import argparse
import sys
import os
import mlflow

# --- KONFIGURASI DAGSHUB---
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/treeasania/Model_Telco_Churn.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "treeasania"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2215b248b0302ccb69efecfff0d1d69bae7dceb0"

# ... lanjut ke kode training ...
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

def train_skilled():
    print("=== Memulai Training Skilled (Manual Log + Tuning) ===")

    # 1. Load Data
    try:
        train_df = pd.read_csv("dataset_preprocessing/train_processed.csv")
        test_df = pd.read_csv("dataset_preprocessing/test_processed.csv")
    except FileNotFoundError:
        # Cadangan jika dijalankan dari dalam folder Membangun_model
        train_df = pd.read_csv("dataset_preprocessing/train_processed.csv")
        test_df = pd.read_csv("dataset_preprocessing/test_processed.csv")

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    # 2. Setup MLflow
    mlflow.set_experiment("Eksperimen_Skilled_Telco")

    # PENTING: Jangan panggil mlflow.autolog() di sini! (Syarat Skilled)

    with mlflow.start_run(run_name="Skilled_XGBoost_Tuning"):
        # 3. Hyperparameter Tuning (Syarat Skilled)
        print("Sedang melakukan GridSearch...")
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        
        # GridSearch
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best Params ditemukan: {best_params}")

        # 4. Evaluasi & Hitung Metriks Manual
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Menghitung metriks agar "sama dengan autolog"
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        # 5. Manual Logging (Syarat Skilled)
        # Log Parameter Terbaik
        mlflow.log_params(best_params)
        
        # Log Metriks
        mlflow.log_metrics(metrics)
    
        print("Menyimpan model ke folder 'model'...")
        import mlflow.sklearn 
        mlflow.xgboost.log_model(best_model, "model")

        print("Training Skilled Selesai. Metriks tercatat manual di MLflow.")

if __name__ == "__main__":
    print("Memulai Training via MLflow Project...")
    
    # 1. Menangkap parameter dari perintah luar (MLProject)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    # 2. Cetak parameter biar kita tahu ini jalan
    print(f"Parameter diterima: n_estimators={args.n_estimators}, learning_rate={args.learning_rate}")

    # 3. Jalankan fungsi training utama kamu
    # Pastikan nama fungsinya sesuai dengan yang ada di script kamu (misal: train_skilled)
    try:
        train_skilled() 
    except NameError:
        # Jaga-jaga kalau nama fungsinya beda
        print("⚠️ Warning: Pastikan nama fungsi training kamu benar (misal: train_skilled())")
