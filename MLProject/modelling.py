import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

DAGSHUB_USERNAME = "anggazakariiya"
REPO_NAME = "Eksperimen_SML_Angga_Zakariya"

def main():
    print("Menginisialisasi DagsHub...")

    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Eksperimen_Diabetes_Advance")

    print("Loading data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'diabetes_preprocessing.csv')
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File diabetes_preprocessing.csv tidak ditemukan.")
        return
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Memulai Tuning Hyperparameter...")
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Mencari parameter terbaik
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Params found: {best_params}")

    # Matikan autolog agar logging murni manual
    mlflow.sklearn.autolog(disable=True)

    print("Mengirim log ke DagsHub...")
    with mlflow.start_run(run_name="Advance_Tuning_Angga"):
        
        # A. Log Hyperparameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # B. Log Metrics
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        print(f"   Metrics -> Acc: {acc:.4f}, F1: {f1:.4f}")
        
        # C. Log Model ke DagsHub
        mlflow.sklearn.log_model(best_model, "model_rf_tuned")
        
        # D. Simpan Model Lokal (Jembatan Deployment)
        local_model_path = os.path.join(script_dir, "model_diabetes.pkl")
        joblib.dump(best_model, local_model_path)
        print(f"💾 Model fisik disimpan di: {local_model_path}")
        

        # Artefak Tambahan
        
        # Confusion Matrix Image
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        
        mlflow.log_artifact("confusion_matrix.png") # Upload
        print("Artefak 1 (Confusion Matrix) terupload.")
        
        #  ROC Curve Image
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        plt.close()
        
        mlflow.log_artifact("roc_curve.png") # Upload
        print("Artefak 2 (ROC Curve) terupload.")
        
        # Bersihkan file sampah
        if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")
        if os.path.exists("roc_curve.png"): os.remove("roc_curve.png")

    print("\nSELESAI! Cek hasil di: https://dagshub.com/" + DAGSHUB_USERNAME + "/" + REPO_NAME)

if __name__ == "__main__":
    main()