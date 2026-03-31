import sys
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
if hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except: pass

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import shap
import warnings

warnings.filterwarnings('ignore')

try:
    suppression_model = joblib.load('models/suppression_model.pkl')
    tox_vec = joblib.load('models/tox_vectorizer.pkl')
    tox_model = joblib.load('models/toxicity_model.pkl')
    tox_cols = joblib.load('models/tox_cols.pkl')
    
    spam_vec = joblib.load('models/spam_vectorizer.pkl')
    spam_model = joblib.load('models/spam_model.pkl')
    
    saf_vec = joblib.load('models/safety_vectorizer.pkl')
    saf_model = joblib.load('models/safety_model.pkl')
    saf_classes = joblib.load('models/safety_classes.pkl')
    suppression_features = joblib.load('models/suppression_features.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    suppression_model = None

api_app = FastAPI(title="SafeSpeak Research API")

class PostRequest(BaseModel):
    text: str
    hashtags: List[str]
    likes: int
    watch_time: float = 50.0

@api_app.post("/predict-post")
def predict_post(req: PostRequest):
    if suppression_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
    
    # Text + Hashtags combined
    full_text = str(req.text)
    if req.hashtags:
        full_text += " " + " ".join(req.hashtags)
        
    # 1. Tox probabilities
    X_tox = tox_vec.transform([full_text])
    tox_probs = tox_model.predict_proba(X_tox)[0]
    
    # 2. Spam probability
    X_spam = spam_vec.transform([full_text])
    spam_prob = float(spam_model.predict_proba(X_spam)[0, 1])
    
    # 3. Safety probabilities
    X_saf = saf_vec.transform([full_text])
    saf_probs = saf_model.predict_proba(X_saf)[0]
    
    # 4. Engagement Score
    impressions = req.likes * 10
    engagement_ratio = float(req.likes) / max(impressions, 1)
    
    # 5. Create DataFrame row
    row_data = list(tox_probs) + [spam_prob] + list(saf_probs) + [engagement_ratio, float(req.watch_time)]
    features = pd.DataFrame([row_data], columns=suppression_features)
    
    # Meta-prediction
    pred = suppression_model.predict(features)[0]
    confidences = suppression_model.predict_proba(features)[0]
    pred_idx = list(suppression_model.classes_).index(pred)
    conf = float(confidences[pred_idx])
    if conf > 0.90:
        conf = 0.90
    
    # Logical Consistency Post-Processing
    safe_idx = list(saf_classes).index('safe')
    sf_probs_unsafe = [p for j, p in enumerate(saf_probs) if j != safe_idx]
    max_saf = max(sf_probs_unsafe) if sf_probs_unsafe else 0
    max_tox = max(tox_probs)
    
    if max_saf > 0.60:
        pred = "Policy-sensitive content"
    if spam_prob > 0.7 and pred == "Normal visibility":
        pred = "Spam-like content"
        
    if pred == "Normal visibility" and (max_tox > 0.7 or spam_prob > 0.7):
        # Override to dominant risk class
        if spam_prob > max_tox:
            pred = "Spam-like content"
        else:
            pred = "Policy-sensitive content"
            
    # SHAP Extract: TreeExplainer doesn't work natively on CalibratedClassifierCV
    # Use the underlying Random Forest of the first calibrated fold as a strong proxy for tree values
    base_rf = suppression_model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_rf)
    shap_vals = explainer.shap_values(features)
    
    # shap_vals shape handles multi-class based on underlying class indices of RF
    rf_pred_idx = list(base_rf.classes_).index(pred) if pred in base_rf.classes_ else pred_idx
    if isinstance(shap_vals, list):
        local_shap = shap_vals[rf_pred_idx][0]
    else:
        # shap >= 0.39 for mult-class
        if len(shap_vals.shape) == 3:
            local_shap = shap_vals[0, :, rf_pred_idx]
        else:
            local_shap = shap_vals[0]
            
    # Generate full raw SHAP data for plotting on frontend
    shap_data = []
    for i, val in enumerate(local_shap):
        if abs(val) > 0.0001:
            feat_name = suppression_features[i].replace('_', ' ').capitalize().replace("prob", "probability")
            shap_data.append({"feature": feat_name, "value": float(val)})
            
    # Sort by absolute impact
    shap_data = sorted(shap_data, key=lambda x: abs(x["value"]), reverse=True)
        
    # Format expected value correctly (could be array or scalar depending on tree model type)
    if isinstance(explainer.expected_value, list) or isinstance(explainer.expected_value, np.ndarray):
        if len(explainer.expected_value) > rf_pred_idx:
            base_val = float(explainer.expected_value[rf_pred_idx])
        else:
            base_val = float(explainer.expected_value[0])
    else:
        base_val = float(explainer.expected_value)

    return {
        "reason": pred,
        "confidence": round(conf, 4),
        "explanation": "SHAP enabled.",
        "shap_data": shap_data,
        "base_value": base_val
    }


class CommentRequest(BaseModel):
    text: str

@api_app.post("/predict-comment")
def predict_comment(req: CommentRequest):
    # Standalone toxicity evaluating pipeline
    if tox_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
        
    text_str = str(req.text)
    X_tox = tox_vec.transform([text_str])
    tox_probs = tox_model.predict_proba(X_tox)[0]
    
    probs_dict = {col: float(prob) for col, prob in zip(tox_cols, tox_probs)}
    neutral_prob = max(1.0 - max(tox_probs), 0.0)
    probs_dict['neutral'] = float(neutral_prob)
    
    dominant_label = max(probs_dict, key=probs_dict.get)
    
    # Generate SHAP explanation for the comment using LinearExplainer
    explainer = shap.LinearExplainer(tox_model.estimators_[list(probs_dict.keys()).index(dominant_label)] if dominant_label != 'neutral' else tox_model.estimators_[0], tox_vec.transform([""]))
    shap_vals = explainer.shap_values(X_tox)[0]
    
    feature_names = tox_vec.get_feature_names_out()
    
    shap_data = []
    # Only keep non-zero features
    for i, val in enumerate(shap_vals):
        if abs(val) > 0.0001:
            shap_data.append({"feature": feature_names[i], "value": float(val)})
            
    # Sort by absolute value for the top chart
    shap_data = sorted(shap_data, key=lambda x: abs(x["value"]), reverse=True)[:15] # Top 15 words
    
    return {
        "dominant_label": dominant_label,
        "probabilities": probs_dict,
        "shap_data": shap_data,
        "base_value": float(explainer.expected_value)
    }

@api_app.post("/predict-post-image")
async def predict_post_image(file: UploadFile = File(...)):
    if suppression_model is None:
        raise HTTPException(status_code=500, detail="Suppression model not loaded.")
        
    try:
        from utils.ocr_extractor import extract_features_from_image
        image_bytes = await file.read()
        extracted_data = extract_features_from_image(image_bytes)
    except Exception as e:
        safe_error_str = str(e).encode('ascii', 'replace').decode('ascii')
        raise HTTPException(status_code=400, detail=f"Failed to process image: {safe_error_str}")
        
    watch_time_est = extracted_data.get("watch_time", 50.0)
    req = PostRequest(
        text=extracted_data.get("text", ""),
        hashtags=extracted_data.get("hashtags", []),
        likes=extracted_data.get("likes", 0),
        watch_time=watch_time_est
    )
    
    result = predict_post(req)
    result["extracted_data"] = extracted_data
    return result

if __name__ == "__main__":
    uvicorn.run("run_api:api_app", host="0.0.0.0", port=8000, reload=False)
