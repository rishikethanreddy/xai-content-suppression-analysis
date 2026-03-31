# Explainable Multi-Label Toxic Comment Detection System

This project is a comprehensive Machine Learning web application designed to detect and explain multiple forms of toxicity in text comments. It utilizes a hybrid feature engineering approach, combining TF-IDF with custom linguistic features, and trains an XGBoost model using a One-vs-Rest strategy. The project integrates SHAP (SHapley Additive exPlanations) to provide deep interpretability and explainability for every prediction.

## Why XAI is Required in Our System

Our project predicts:
- Whether content will be suppressed/flagged 
- The reason for suppression
- The probability/confidence score

Since content suppression affects user visibility, freedom of expression, and platform trust, it is critical that the system does not behave like a black-box model.

Without explainability:
Input text → Model → Suppressed (87%)
The user does not know why.

Therefore, we integrate SHAP (SHapley Additive exPlanations) to ensure transparency and interpretability.

### 🔍 What is SHAP?
SHAP is an Explainable AI technique based on Shapley values from cooperative game theory.
In game theory:
- Each player contributes to the total outcome.
- Shapley value calculates each player's fair contribution.

In our ML system:
- Each word in the text is treated like a “player”.
- The prediction (suppressed/not suppressed) is the “game outcome”.
- SHAP calculates how much each word contributed to that prediction.

### ⚙️ How SHAP Works in Our Model
Let’s assume the model predicts:
Suppressed = 0.82 probability

SHAP breaks this prediction into:
**Final Prediction = Base Value + Sum of Feature Contributions**

**🔹 Base Value**
The average suppression probability across the dataset.
*Example: Base Value = 0.40*
This means: If no specific text features were known, the average suppression risk is 40%.

**🔹 Feature Contributions (SHAP Values)**
Suppose input text: *“You are disgusting and should be banned”*
SHAP might produce:
| Word | SHAP Contribution |
|---|---|
| disgusting | +0.22 |
| banned | +0.15 |
| you | +0.03 |
| are | -0.01 |

Now:
Final Prediction = 0.40 + (0.22 + 0.15 + 0.03 - 0.01) ≈ 0.79

This shows:
- “disgusting” strongly increased suppression probability
- Neutral words had minimal effect

### 📊 Types of Explanation in Our Project

**1️⃣ Local Explanation (Per Input)**
For every user input, SHAP:
- Identifies top contributing words
- Shows positive contributors (increase suppression risk)
- Shows negative contributors (reduce suppression risk)

We present:
- Top positive/negative features
- Visual human-readable explanations directly in the UI

*Example: “The suppression prediction was mainly influenced by aggressive terms such as ‘disgusting’ (+0.22) and ‘banned’ (+0.15).”*
This ensures transparency at the individual decision level.

**2️⃣ Global Explanation (Model-Level)**
SHAP also provides a summary plot showing:
- Which features generally influence suppression decisions
- Most impactful words across the dataset

This helps us understand model behavior, bias patterns, and important moderation indicators.

## Project Structure

```
project/
│
├── data/
│     └── train.csv                 # Raw dataset
├── plots/                          # Generated evaluation and EDA plots
├── model/
│     └── xgboost_model.pkl         # Trained model pipeline
├── feature_engineering.py          # Preprocessing and feature engineering
├── train_model.py                  # Model training and evaluation
├── app.py                          # Streamlit web interface
├── vectorizer.pkl                  # Saved TF-IDF vectorizer
├── feature_names.pkl               # Combined feature names list
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

## Step-by-Step Instructions to Run Locally

Follow these steps to set up, train, and run the complete system.

### Step 1: Install Dependencies
Ensure you are using Python 3.10+. Open your terminal and run:
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
Ensure that your `train.csv` is located in the `data/` directory. Create the `data` directory if it does not exist:
```bash
mkdir -p data
# Place train.csv into the data/ folder
```

### Step 3: Train the Model and Generate Outputs
Run the `train_model.py` script. This script executes the entire machine learning pipeline, printing extensive academic evaluation metrics to the console and saving visualizations to the `plots/` directory.

```bash
python train_model.py
```

**What this does:**
1. Loads and explores the data.
2. Extracts TF-IDF and custom linguistic features.
3. Splits into training and validation sets.
4. Trains the `OneVsRestClassifier(XGBClassifier)`.
5. Prints evaluation metrics (Micro/Macro F1, Hamming Loss, ROC-AUC per label, Classification Report).
6. Generates ROC Curves, Precision-Recall Curves, and Confusion Matrices.
7. Saves the trained model and vectorizer.

### Step 4: Run the Web Application
Start the Streamlit interface to interact with the trained model and view SHAP explanations for custom inputs:

```bash
streamlit run app.py
```

Open the provided Local URL (usually `http://localhost:8501`) in your web browser. 
- Enter a comment in the text box (e.g., "You are a stupid idiot").
- Click Predict to view probabilities across the 6 toxicity categories.
- Review the SHAP Explainability section to see the exact words and features driving the prediction.
