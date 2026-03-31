import pandas as pd
import numpy as np
import re
import os
import joblib
import urllib.request
import zipfile
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

os.makedirs('models', exist_ok=True)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s!\?]', '', text)
    return text.strip()

def train_toxicity():
    print("--- Training Toxicity Model ---")
    df = pd.read_csv('train.csv')
    df['clean_text'] = df['comment_text'].apply(clean_text)
    
    # Map severe_toxic -> very_toxic, identity_hate -> hate
    df.rename(columns={'severe_toxic': 'very_toxic', 'identity_hate': 'hate'}, inplace=True)
    target_cols = ['toxic', 'very_toxic', 'obscene', 'threat', 'insult', 'hate']
    
    # Sample down to 30k for speed
    df_sample = df.sample(n=30000, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(df_sample['clean_text'], df_sample[target_cols], test_size=0.1, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced'))
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_val_vec)
    y_prob = model.predict_proba(X_val_vec)
    
    print(f"Micro F1: {f1_score(y_val, y_pred, average='micro'):.4f}")
    print(f"Macro F1: {f1_score(y_val, y_pred, average='macro'):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_prob, average='macro'):.4f}")
    
    joblib.dump(vectorizer, 'models/tox_vectorizer.pkl')
    joblib.dump(model, 'models/toxicity_model.pkl')
    # Save target cols order
    joblib.dump(target_cols, 'models/tox_cols.pkl')
    print("Saved Toxicity Model.\n")

def train_spam():
    print("--- Training Spam Model ---")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    resp = urllib.request.urlopen(url)
    zipfile.ZipFile(BytesIO(resp.read())).extractall("spam_data")
    
    # Read spam data
    df = pd.read_csv("spam_data/SMSSpamCollection", sep='\t', names=['label', 'text'])
    df['clean_text'] = df['text'].apply(clean_text)
    df['is_spam'] = (df['label'] == 'spam').astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(df['clean_text'], df['is_spam'], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_val_vec)
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")
    
    joblib.dump(vectorizer, 'models/spam_vectorizer.pkl')
    joblib.dump(model, 'models/spam_model.pkl')
    print("Saved Spam Model.\n")

def train_safety():
    print("--- Training Safety Sensitivity Model (V4) ---")
    # Synthesize data for: sexual_violence, self_harm, exploitation_threats, criminal_investigation, disturbing_content, general_violence, safe
    data = []
    
    # sexual_violence
    sv_words = ['rape', 'assault', 'harass', 'groped', 'forced', 'violation', 'nonconsensual', 'molest', 'abuse']
    for _ in range(500):
        words = np.random.choice(sv_words, 4).tolist() + ['he', 'did', 'this']
        data.append((" ".join(words), 'sexual_violence'))
        
    # self_harm
    sh_words = ['suicide', 'kill', 'myself', 'cut', 'depressed', 'end', 'life', 'wrist', 'bleed', 'die']
    for _ in range(500):
        words = np.random.choice(sh_words, 4).tolist() + ['want', 'to']
        data.append((" ".join(words), 'self_harm'))
        
    # exploitation_threats
    exp_words = ['leak', 'nudes', 'expose', 'revenge', 'porn', 'blackmail', 'pictures', 'send', 'money', 'post']
    for _ in range(500):
        words = np.random.choice(exp_words, 4).tolist() + ['i', 'will']
        data.append((" ".join(words), 'exploitation_threats'))
        
    # criminal_investigation
    crim_words = ['police', 'arrest', 'investigation', 'suspect', 'warrant', 'court', 'trial', 'homicide', 'detective', 'evidence']
    for _ in range(500):
        words = np.random.choice(crim_words, 4).tolist() + ['the', 'officer', 'said']
        data.append((" ".join(words), 'criminal_investigation'))
        
    # disturbing_content
    dist_words = ['gory', 'mutilated', 'corpse', 'accident', 'gruesome', 'flesh', 'decapitated', 'blood', 'horrific', 'remains']
    for _ in range(500):
        words = np.random.choice(dist_words, 4).tolist() + ['it', 'was', 'very']
        data.append((" ".join(words), 'disturbing_content'))
        
    # general_violence
    viol_words = ['punch', 'fight', 'stab', 'shoot', 'beat', 'attack', 'gun', 'knife', 'hurt', 'kick']
    for _ in range(500):
        words = np.random.choice(viol_words, 4).tolist() + ['they', 'started', 'to']
        data.append((" ".join(words), 'general_violence'))
        
    # safe
    safe_words = ['flower', 'happy', 'good', 'morning', 'cat', 'dog', 'weather', 'beautiful', 'coffee', 'friend', 'car', 'sports', 'football', 'music']
    for _ in range(1000):
        words = np.random.choice(safe_words, 5).tolist() + ['is', 'very', 'nice']
        data.append((" ".join(words), 'safe'))
        
    df = pd.DataFrame(data, columns=['text', 'label'])
    df['clean_text'] = df['text'].apply(clean_text)
    
    X_train, X_val, y_train, y_val = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_val_vec)
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    
    joblib.dump(vectorizer, 'models/safety_vectorizer.pkl')
    joblib.dump(model, 'models/safety_model.pkl')
    # Save policy classes order
    joblib.dump(model.classes_, 'models/safety_classes.pkl')
    print("Saved Safety Model.\n")

if __name__ == "__main__":
    train_toxicity()
    train_spam()
    train_safety()
