import sys
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

suppression_model = joblib.load('models/suppression_model.pkl')
tox_vec = joblib.load('models/tox_vectorizer.pkl')
tox_model = joblib.load('models/toxicity_model.pkl')
tox_cols = joblib.load('models/tox_cols.pkl')
spam_vec = joblib.load('models/spam_vectorizer.pkl')
spam_model = joblib.load('models/spam_model.pkl')
pol_vec = joblib.load('models/policy_vectorizer.pkl')
pol_model = joblib.load('models/policy_model.pkl')
pol_classes = joblib.load('models/policy_classes.pkl')
suppression_features = joblib.load('models/suppression_features.pkl')

full_text = '2.4K 11 babunuvubtechah A female engineering student in Bachupally was allegedly drugged and sexually assaulted by a fellow student for nearly year: The case came to light after she reportedly attempted suicide: The accused allegedly threatened to leak private photos Family claims college staff ignored earlier complaints_ Police investigation is ongoing:'
X_pol = pol_vec.transform([full_text])
pol_probs = pol_model.predict_proba(X_pol)[0]
print('pol_probs:', list(zip(pol_classes, pol_probs)))

safe_idx = list(pol_classes).index('safe_content')
p_probs_unsafe = [p for j, p in enumerate(pol_probs) if j != safe_idx]
max_pol = max(p_probs_unsafe) if p_probs_unsafe else 0
print('max_pol:', max_pol)
