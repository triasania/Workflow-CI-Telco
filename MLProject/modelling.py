# -*- coding: utf-8 -*-
import os
import shutil
import argparse
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, roc_auc_score, confusion_matrix, roc_curve)
from mlflow.models.signature import infer_signature

# --- 1. KONFIGURASI DAGSHUB ---
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/treeasania/Model_Telco_Churn.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "treeasania"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2215b248b0302ccb69efecfff0d1d69bae7dceb0"

def train_final_submission(n_est_input, lr_input):
    print("=== Memulai Training Advanced (Compatible with CI/CD) ===")

    # --- 2. Load Data ---
    # Cek berbagai kemungkinan lokasi file (Local vs Docker vs Github Actions)
    paths = [
        "MLProject/dataset_preprocessing/train_processed.csv", # Struktur Github Actions
        "dataset_preprocessing/train_processed.csv",           # Struktur Local/Docker
        "Membangun_model/dataset_preprocessing/train_processed.csv" # Struktur Notebook
    ]
    
    train_df = None
    for p in paths:
        if os.path.exists(p):
            train_df = pd.read_csv(p)
            test_df = pd.read_csv(p.replace("train", "test"))
            print(f"Dataset ditemukan di: {p}")
            break
    
    if train_df is None:
        raise FileNotFoundError("Dataset tidak ditemukan di path manapun!")

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    # --- 3. Setup MLflow ---
    #mlflow.set_experiment("Eksperimen_Advanced_Telco")

    # Start Run
    with mlflow.start_run(run_name="Advanced_CI_Pipeline_Run"):
        
        # --- 4. Hyperparameter Tuning (Syarat Advanced) ---
        # Kita gunakan input dari CLI sebagai salah satu opsi grid search
        print("Sedang melakukan GridSearch...")
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        
        param_grid = {
            'n_estimators': [n_est_input, 100], # Gunakan input CLI & default
            'max_depth': [5],
            'learning_rate': [lr_input, 0.1]    # Gunakan input CLI & default
        }
        
        # Hapus duplikat di list jika input sama dengan default
        for k in param_grid:
            param_grid[k] = list(set(param_grid[k]))

        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best Params: {best_params}")

        # --- 5. Evaluasi ---
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "f1_score": f1_score(y_test, y_pred)
        }
        
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        # --- 6. Upload 3 Artefak Gambar (Syarat Advanced) ---
        print("Membuat & Upload 3 Grafik...")
        
        # A. Confusion Matrix
        plt.figure(figsize=(6, 5)); sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix'); plt.savefig("confusion_matrix.png"); plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        # B. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr); plt.title('ROC Curve')
        plt.savefig("roc_curve.png"); plt.close()
        mlflow.log_artifact("roc_curve.png")

        # C. Feature Importance
        plt.figure(figsize=(8, 6))
        sorted_idx = best_model.feature_importances_.argsort()
        plt.barh(X_train.columns[sorted_idx], best_model.feature_importances_[sorted_idx])
        plt.title("Feature Importance"); plt.tight_layout()
        plt.savefig("feature_importance.png"); plt.close()
        mlflow.log_artifact("feature_importance.png")

        # --- 7. UPLOAD MODEL & LOCAL SAVE (PENTING UTK DOCKER) ---
        print("Menyimpan model secara Aman (Lokal + Cloud)...")
        
        # Nama folder lokal yang akan dipakai oleh Docker nanti
        local_path = "model_docker_local"
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
            
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        # A. Simpan ke folder lokal (Untuk keperluan build Docker di step selanjutnya)
        mlflow.xgboost.save_model(best_model, path=local_path, signature=signature)
        print(f"Model tersimpan di folder lokal: {local_path}")
        
        # B. Upload folder lokal tersebut ke DagsHub (Untuk keperluan grading)
        mlflow.log_artifacts(local_path, artifact_path="model")
        
        # C. Register Model
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model_registry_dummy",
            signature=signature,
            registered_model_name="Telco_Churn_Advanced_Final"
        )
        
        print("Upload Model Selesai.")

        # Bersih-bersih file gambar (Folder model jangan dihapus dulu, dipakai Docker)
        for f in ["confusion_matrix.png", "roc_curve.png", "feature_importance.png"]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    # --- ARGPARSE (Supaya bisa dipanggil via Terminal/GitHub Actions) ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    # Jalankan fungsi utama dengan argumen
    train_final_submission(args.n_estimators, args.learning_rate)
