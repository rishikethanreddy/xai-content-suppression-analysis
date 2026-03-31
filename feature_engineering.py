import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from scipy.sparse import hstack
import joblib
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Define custom bad words simply
BANNED_WORDS = set(['stupid', 'idiot', 'dumb', 'shit', 'fuck', 'bitch', 'crap', 'ass', 'asshole', 'dick', 'pussy'])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z0-9\s!\?]', '', text)
    return text.strip()

def extract_custom_features(texts):
    df_features = pd.DataFrame()
    df_features['comment_length'] = texts.apply(lambda x: len(str(x)))
    df_features['word_count'] = texts.apply(lambda x: len(str(x).split()))
    df_features['uppercase_ratio'] = texts.apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1e-5))
    df_features['exclamation_count'] = texts.apply(lambda x: str(x).count('!'))
    df_features['question_count'] = texts.apply(lambda x: str(x).count('?'))
    df_features['sentiment_score'] = texts.apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df_features['profanity_count'] = texts.apply(lambda x: sum(1 for word in str(x).lower().split() if word in BANNED_WORDS))
    df_features['unique_word_ratio'] = texts.apply(lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1e-5))
    return df_features

def run_feature_engineering(csv_path):
    print("\n" + "="*50)
    print("1. DATA UNDERSTANDING OUTPUTS")
    print("="*50)
    
    df = pd.read_csv(csv_path)
    
    print("\n[INFO] First 5 rows of dataset:")
    print(df.head())
    
    print(f"\n[INFO] Dataset shape: {df.shape}")
    
    print(f"\n[INFO] Column names:\n{df.columns.tolist()}")
    
    print(f"\n[INFO] Data types:\n{df.dtypes}")
    
    print(f"\n[INFO] Null value count:\n{df.isnull().sum()}")
    
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    print("\n[INFO] Label distribution for each toxicity category:")
    label_counts = df[target_cols].sum()
    print(label_counts)
    
    # Plot class imbalance graph
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Class Imbalance: Number of Comments per Toxicity Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/class_imbalance.png')
    plt.close()
    print("[INFO] Saved class imbalance graph to 'plots/class_imbalance.png'")
    
    # Preprocessing
    print("\n" + "="*50)
    print("2. PREPROCESSING OUTPUTS")
    print("="*50)
    
    sample_comments = df['comment_text'].head(5).copy()
    print("\n[INFO] 5 original comments:")
    for i, c in enumerate(sample_comments):
        print(f"{i+1}: {c[:100]}...")
        
    df['clean_comment'] = df['comment_text'].apply(clean_text)
    
    print("\n[INFO] 5 comments after cleaning:")
    for i, c in enumerate(df['clean_comment'].head(5)):
        print(f"{i+1}: {c[:100]}...")
        
    comment_lengths = df['clean_comment'].apply(len)
    print(f"\n[INFO] Average comment length: {comment_lengths.mean():.2f} characters")
    print(f"[INFO] Max comment length: {comment_lengths.max()} characters")
    print(f"[INFO] Min comment length: {comment_lengths.min()} characters")
    
    word_counts = df['clean_comment'].apply(lambda x: len(x.split()))
    # Cap at 500 for better visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(word_counts[word_counts <= 500], bins=50, kde=True)
    plt.title('Word Count Distribution (Truncated at 500 words)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('plots/word_count_distribution.png')
    plt.close()
    print("[INFO] Saved word count distribution plot to 'plots/word_count_distribution.png'")
    
    X = df[['comment_text', 'clean_comment']]
    y = df[target_cols]
    
    print("\n" + "="*50)
    print("4. TRAIN-VALIDATION SPLIT")
    print("="*50)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Training set size: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Validation set size: X={X_val.shape}, y={y_val.shape}")
    
    print("\n[INFO] Label distribution in Training set:")
    print(y_train.sum())
    print("\n[INFO] Label distribution in Validation set:")
    print(y_val.sum())
    
    print("\n" + "="*50)
    print("3. FEATURE ENGINEERING OUTPUTS")
    print("="*50)
    
    # TF-IDF
    print("\n[INFO] Extracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english', lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train['clean_comment'])
    X_val_tfidf = vectorizer.transform(X_val['clean_comment'])
    
    vocab_size = len(vectorizer.vocabulary_)
    print(f"[INFO] Vocabulary size: {vocab_size}")
    print(f"[INFO] Shape of TF-IDF matrix (Train): {X_train_tfidf.shape}")
    
    # Top 20 important TF-IDF words based on IDF scores
    idf_scores = vectorizer.idf_
    indices = np.argsort(idf_scores)[:20]
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_words = feature_names[indices]
    print(f"[INFO] Top 20 important TF-IDF words (lowest IDF):\n{top_words}")
    
    # Engineered Features
    print("\n[INFO] Extracting Engineered Features...")
    X_train_custom = extract_custom_features(X_train['comment_text'])
    X_val_custom = extract_custom_features(X_val['comment_text'])
    
    print("\n[INFO] First 5 rows of engineered features:")
    display_cols = ['comment_length', 'word_count', 'uppercase_ratio', 'exclamation_count', 'sentiment_score', 'profanity_count']
    print(X_train_custom[display_cols].head())
    
    X_train_combined = hstack([X_train_tfidf, X_train_custom.values])
    X_val_combined = hstack([X_val_tfidf, X_val_custom.values])
    
    print(f"\n[INFO] Final combined feature matrix shape (Train): {X_train_combined.shape}")
    print(f"[INFO] Final combined feature matrix shape (Val): {X_val_combined.shape}")
    
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("[INFO] Vectorizer saved to 'vectorizer.pkl'")
    
    # Also save the feature names list for SHAP
    custom_feat_names = X_train_custom.columns.tolist()
    all_feature_names = list(feature_names) + custom_feat_names
    joblib.dump(all_feature_names, 'feature_names.pkl')
    
    return X_train_combined, X_val_combined, y_train, y_val, all_feature_names

if __name__ == "__main__":
    run_feature_engineering('data/train.csv')
