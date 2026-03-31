import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def synthesize_suppression_dataset():
    print("Loading base models...")
    tox_vec = joblib.load('models/tox_vectorizer.pkl')
    tox_model = joblib.load('models/toxicity_model.pkl')
    
    spam_vec = joblib.load('models/spam_vectorizer.pkl')
    spam_model = joblib.load('models/spam_model.pkl')
    
    saf_vec = joblib.load('models/safety_vectorizer.pkl')
    saf_model = joblib.load('models/safety_model.pkl')
    saf_classes = joblib.load('models/safety_classes.pkl')
    
    # Load some real texts
    df = pd.read_csv('train.csv').sample(n=10000, random_state=42)
    texts = df['comment_text'].fillna("").astype(str).tolist()
    
    print("Computing base model probabilities...")
    # Tox
    X_tox = tox_vec.transform(texts)
    tox_probs = tox_model.predict_proba(X_tox) # shape (10000, 6)
    
    # Spam
    X_spam = spam_vec.transform(texts)
    spam_probs = spam_model.predict_proba(X_spam)[:, 1] # shape (10000,)
    
    # Safety
    X_saf = saf_vec.transform(texts)
    saf_probs = saf_model.predict_proba(X_saf) # shape (10000, 7)
    
    # Engagement
    np.random.seed(42)
    impressions = np.random.lognormal(mean=8.0, sigma=1.5, size=10000).astype(int) + 100
    like_ratios = np.random.beta(a=2, b=10, size=10000)
    likes = (impressions * like_ratios).astype(int)
    engagement_ratios = likes / np.maximum(impressions, 1)
    watch_time_percents = np.random.uniform(0, 100, size=10000)
    
    # Construct feature matrix
    X_features = []
    labels = []
    
    safe_idx = list(saf_classes).index('safe')
    
    for i in range(10000):
        t_probs = tox_probs[i]
        s_prob = spam_probs[i]
        sf_probs = saf_probs[i]
        eng_ratio = engagement_ratios[i]
        wt_pct = watch_time_percents[i]
        
        row = list(t_probs) + [s_prob] + list(sf_probs) + [eng_ratio, wt_pct]
        X_features.append(row)
        
        # Weak supervision labeling rule
        max_tox = np.max(t_probs)
        sf_probs_unsafe = [p for j, p in enumerate(sf_probs) if j != safe_idx]
        max_saf = np.max(sf_probs_unsafe) if sf_probs_unsafe else 0
        
        rand_noise = np.random.rand()
        
        if max_saf > 0.60:
            labels.append("Policy-sensitive content")
        elif max_tox > 0.5 or (max_tox > 0.3 and rand_noise > 0.8):
            labels.append("Policy-sensitive content")
        elif s_prob > 0.5 or (s_prob > 0.3 and rand_noise > 0.8):
            labels.append("Spam-like content")
        elif eng_ratio < 0.05 or wt_pct < 20.0:
            labels.append("Low engagement")
        else:
            labels.append("Normal visibility")
            
    feature_names = ['toxic', 'very_toxic', 'obscene', 'threat', 'insult', 'hate', 
                     'spam_prob'] + list(saf_classes) + ['engagement_ratio', 'watch_time_percent']
                     
    X_df = pd.DataFrame(X_features, columns=feature_names)
    y_series = pd.Series(labels)
    
    return X_df, y_series
    

def main():
    X, y = synthesize_suppression_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest with Calibration...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    # Using 'sigmoid' method of calibration as 'isotonic' may overfit on 10k items
    calibrated_rf = CalibratedClassifierCV(rf, cv=5, method='sigmoid')
    calibrated_rf.fit(X_train, y_train)
    
    y_pred = calibrated_rf.predict(X_test)
    
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(calibrated_rf, 'models/suppression_model.pkl')
    joblib.dump(X_train.columns.tolist(), 'models/suppression_features.pkl')
    print("Saved Calibrated Suppression Meta-Classifier to models/suppression_model.pkl")

if __name__ == "__main__":
    main()
