# Importing necessary libraries and modules
import os
import ssl
import json
import warnings
warnings.filterwarnings('ignore')

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix, roc_curve, precision_recall_curve

def main():
    # Defining paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = CURRENT_DIR if os.path.basename(CURRENT_DIR) != "model" else os.path.dirname(CURRENT_DIR)
    MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Importing the dataset from UCI ML Repository
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features.copy()
    y_raw = bank_marketing.data.targets.copy()

    target_col = 'y'
    y = (y_raw[target_col].astype(str).str.strip().str.lower() == 'yes').astype(int)
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
    
    # Stratified Train Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101, stratify=y)
    
    # Saveing test_data.csv
    test_df = X_test.copy()
    test_df[target_col] = y_raw.loc[X_test.index, target_col].values
    test_data_path = os.path.join(PROJECT_ROOT, "test_data.csv")
    test_df.to_csv(test_data_path, index=False)
    
    # Feature Preprocessing Pipeline
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    
    # Saving preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    
    # Hyperparameter Tuning & Model Training
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=101)
    
    models_config = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=102),
            "param_grid": {
                "C": [0.1, 1.0, 5.0],
                "class_weight": ["balanced", None]
            },
            "scoring": "f1",
            "filename": "model_logistic_regression.joblib"
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=102),
            "param_grid": {
                "max_depth": [7, 10, 15],
                "min_samples_split": [10, 20],
                "min_samples_leaf": [5, 10],
                "class_weight": ["balanced", None]
            },
            "scoring": "f1",
            "filename": "model_decision_tree.joblib"
        },
        "KNN": {
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "param_grid": {
                "n_neighbors": [5, 7, 9],
                "weights": ["uniform", "distance"]
            },
            "scoring": "f1",
            "filename": "model_knn.joblib"
        },
        "Naive Bayes": {
            "estimator": GaussianNB(),
            "param_grid": {
                "var_smoothing": [1e-9, 1e-5, 1e-3, 0.01, 0.1]
            },
            "scoring": "f1",
            "filename": "model_naive_bayes.joblib"
        },
        "Support Vector Machine": {
            "estimator": SVC(kernel='rbf', probability=True, random_state=102),
            "param_grid": {
                "C": [1.0],
                "class_weight": ["balanced"]
            },
            "scoring": "f1",
            "filename": "model_svm.joblib"
        }
    }
    
    # Training and evaluating each model
    metrics_summary = {}
    comparison_table = []
    for name, config in models_config.items():
        print(f"Training and evaluating model: {name}")
        base_clf = config["estimator"]
        param_grid = config["param_grid"]
        filename = config["filename"]
        scoring = config["scoring"]
        model_save_path = os.path.join(MODEL_DIR, filename)
         
        grid = GridSearchCV(base_clf, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid.fit(X_train_proc, y_train)
        best_params = grid.best_params_
        
        best_clf = grid.best_estimator_
        best_clf.fit(X_train_proc, y_train)
                
        # Inference and Metrics Calculation
        y_pred = best_clf.predict(X_test_proc)
        if hasattr(best_clf, "predict_proba"):
            y_proba = best_clf.predict_proba(X_test_proc)[:, 1]
        elif hasattr(best_clf, "decision_function"):
            y_proba = best_clf.decision_function(X_test_proc)
        else:
            y_proba = y_pred
        
        # Calculate all 6 mandatory metrics at standard threshold (t=0.50)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Threshold Optimization Analysis across [0.10, 0.90]
        threshold_candidates = np.linspace(0.10, 0.90, 81)
        best_t_f1 = 0.50
        max_f1_score = 0.0
        best_t_mcc = 0.50
        max_mcc_score = -1.0
        
        for t in threshold_candidates:
            y_pred_t = (y_proba >= t).astype(int)
            t_f1 = f1_score(y_test, y_pred_t, zero_division=0)
            t_mcc = matthews_corrcoef(y_test, y_pred_t)
            if t_f1 > max_f1_score:
                max_f1_score = t_f1
                best_t_f1 = t
            if t_mcc > max_mcc_score:
                max_mcc_score = t_mcc
                best_t_mcc = t
                
        # Metrics at optimal F1 threshold
        y_pred_opt = (y_proba >= best_t_f1).astype(int)
        opt_acc = accuracy_score(y_test, y_pred_opt)
        opt_prec = precision_score(y_test, y_pred_opt, zero_division=0)
        opt_rec = recall_score(y_test, y_pred_opt, zero_division=0)
        opt_f1 = f1_score(y_test, y_pred_opt, zero_division=0)
        opt_mcc = matthews_corrcoef(y_test, y_pred_opt)
        
        # Youden J statistic (ROC Operating Point)
        fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
        j_scores = tpr - fpr
        best_idx_j = int(np.argmax(j_scores))
        best_t_youden = float(roc_thresh[best_idx_j])
        
        # Save model artifact
        joblib.dump(best_clf, model_save_path)
        file_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
        
        # JSON serialization of hyperparameters
        clean_params = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in best_params.items()}
        
        metrics_summary[name] = {
            "Accuracy": round(float(acc), 4),
            "AUC": round(float(auc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
            "MCC": round(float(mcc), 4),
            "Standard_Threshold": 0.50,
            "Optimal_Threshold_F1": round(float(best_t_f1), 2),
            "Optimal_Threshold_MCC": round(float(best_t_mcc), 2),
            "Youden_J_Threshold": round(float(best_t_youden), 4),
            "Metrics_At_Optimal_Threshold": {
                "Accuracy": round(float(opt_acc), 4),
                "Precision": round(float(opt_prec), 4),
                "Recall": round(float(opt_rec), 4),
                "F1": round(float(opt_f1), 4),
                "MCC": round(float(opt_mcc), 4)
            },
            "Best_Hyperparameters": clean_params,
            "CV_Score": round(float(grid.best_score_), 4),
            "Model_File": filename,
            "Confusion_Matrix": cm
        }
        
        comparison_table.append({
            "ML Model Name": name,
            "Accuracy": f"{acc:.4f}",
            "AUC": f"{auc:.4f}",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "F1": f"{f1:.4f}",
            "MCC": f"{mcc:.4f}"
        })
        
    # Save metrics summary to JSON
    summary_path = os.path.join(MODEL_DIR, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

if __name__ == "__main__":
    main()
