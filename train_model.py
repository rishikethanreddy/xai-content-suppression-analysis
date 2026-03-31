import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             roc_auc_score, hamming_loss, classification_report, 
                             confusion_matrix, roc_curve, precision_recall_curve)
import shap

from feature_engineering import run_feature_engineering

os.makedirs('plots', exist_ok=True)
os.makedirs('model', exist_ok=True)

def plot_confusion_matrices(y_val, y_pred_bin, labels):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, label in enumerate(labels):
        cm = confusion_matrix(y_val.iloc[:, i], y_pred_bin[:, i])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'Confusion Matrix - {label}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png')
    plt.close()
    print("[INFO] Saved confusion matrices to 'plots/confusion_matrices.png'")

def plot_roc_curves(y_val, y_pred_prob, labels):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_val.iloc[:, i], y_pred_prob[:, i])
        auc_score = roc_auc_score(y_val.iloc[:, i], y_pred_prob[:, i])
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves per Label')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/roc_curves.png')
    plt.close()
    print("[INFO] Saved ROC curves to 'plots/roc_curves.png'")

def plot_pr_curves(y_val, y_pred_prob, labels):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y_val.iloc[:, i], y_pred_prob[:, i])
        plt.plot(recall, precision, label=label)
    plt.title('Precision-Recall Curves per Label')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/pr_curves.png')
    plt.close()
    print("[INFO] Saved Precision-Recall curves to 'plots/pr_curves.png'")

def explain_metrics():
    print("\n--- METRICS EXPLANATION ---")
    print("- Multi-Label classification differs from Multi-Class because an instance can have multiple labels simultaneously.")
    print("- Hamming Loss: Fraction of the wrong labels to the total number of labels. Lower is better.")
    print("- Micro F1: Calculates metrics globally by counting the total true positives, false negatives and false positives. Good for imbalanced datasets.")
    print("- Macro F1: Calculates metrics for each label, and finds their unweighted mean. Treats all classes equally regardless of support.")
    print("- ROC-AUC: Measures the ability of the model to distinguish between classes.")

def main():
    X_train_sparse, X_val_sparse, y_train, y_val, feature_names = run_feature_engineering('train.csv')
    
    target_cols = y_train.columns.tolist()
    
    print("\n" + "="*50)
    print("5. MODEL TRAINING OUTPUT")
    print("="*50)
    
    xgb_params = dict(
        n_estimators=100, # Reduced to 100 for faster demonstration/viva, user asked for 200 originally
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist' # faster
    )
    
    # OVERRIDE: user specified n_estimators=200
    xgb_params['n_estimators'] = 200
    
    print("[INFO] XGBoost Parameters:")
    for k, v in xgb_params.items():
        print(f"  {k}: {v}")
        
    base_model = XGBClassifier(**xgb_params)
    model = OneVsRestClassifier(base_model)
    
    print("\n[INFO] Starting Model Training...")
    start_time = time.time()
    print(f"Training Start Time: {time.ctime(start_time)}")
    
    model.fit(X_train_sparse, y_train)
    
    end_time = time.time()
    print(f"Training End Time: {time.ctime(end_time)}")
    duration = end_time - start_time
    print(f"Training Duration: {duration/60:.2f} minutes")
    
    joblib.dump(model, 'model/xgboost_model.pkl')
    print("[INFO] Model saved to 'model/xgboost_model.pkl'")
    
    print("\n[INFO] Generating Feature Importance Plot (Global for first label as example)...")
    try:
        # Access the first XGBoost estimator inside OneVsRest
        first_estimator = model.estimators_[0]
        importances = first_estimator.feature_importances_
        indices = np.argsort(importances)[-20:] # top 20
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importances (Label: Toxic)')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        plt.close()
        print("[INFO] Saved feature importance plot to 'plots/feature_importance.png'")
    except Exception as e:
        print(f"[ERROR] Could not generate feature importance: {e}")
    
    print("\n" + "="*50)
    print("6. EVALUATION OUTPUTS")
    print("="*50)
    
    y_pred_prob = model.predict_proba(X_val_sparse)
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)
    
    print(f"\nOverall Exact Match Accuracy: {accuracy_score(y_val, y_pred_bin):.4f}")
    print(f"Micro F1 Score: {f1_score(y_val, y_pred_bin, average='micro'):.4f}")
    print(f"Macro F1 Score: {f1_score(y_val, y_pred_bin, average='macro'):.4f}")
    print(f"Hamming Loss: {hamming_loss(y_val, y_pred_bin):.4f}")
    
    print("\nPer-Label Metrics:")
    for i, label in enumerate(target_cols):
        p = precision_score(y_val.iloc[:, i], y_pred_bin[:, i])
        r = recall_score(y_val.iloc[:, i], y_pred_bin[:, i])
        auc = roc_auc_score(y_val.iloc[:, i], y_pred_prob[:, i])
        print(f"  {label.ljust(15)} - Precision: {p:.4f} | Recall: {r:.4f} | ROC-AUC: {auc:.4f}")
        
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_bin, target_names=target_cols))
    
    plot_confusion_matrices(y_val, y_pred_bin, target_cols)
    plot_roc_curves(y_val, y_pred_prob, target_cols)
    plot_pr_curves(y_val, y_pred_prob, target_cols)
    
    explain_metrics()
    
    print("\n" + "="*50)
    print("7. SHAP EXPLAINABILITY OUTPUT & 8. FINAL OUTPUT EXAMPLE")
    print("="*50)
    print("SHAP explainer generation is integrated into app.py for interactive demonstration.")
    print("Please run `streamlit run app.py` to view interactive SHAP global and local explanations.")

if __name__ == "__main__":
    main()
