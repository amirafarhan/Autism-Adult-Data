# 🧩 Autism Screening Adult Data — Analysis & Explorer

Data analysis of the **Autism Screening Adult** dataset (UCI Machine Learning Repository, 704 adults): data
cleaning, exploratory analysis, classification models, and an interactive **Streamlit** dashboard.

> **Not a diagnostic tool.** The AQ-10 is a short screening questionnaire. Only a qualified clinician can assess autism.

## 📊 Dataset

Each row is one respondent:

- `A1_Score` … `A10_Score` — the ten AQ-10 screening answers, scored 0/1
- `result` — the AQ-10 score (0–10)
- demographics: age, gender, ethnicity, born with jaundice, family member with autism, country of residence,
  who completed the test
- `class/ASD` — the screening outcome (YES / NO)

## 🔍 Key finding: the label is rule-based

Checked on all 704 rows:

| Check | Result |
|---|---|
| `result` equals the sum of `A1`…`A10` | 100% of rows |
| `class/ASD` is `YES` exactly when `result ≥ 7` (the AQ-10 cut-off) | 100% of rows |

So any model given `result` (or the ten item scores) simply rediscovers the cut-off, which is why the notebook's
Logistic Regression, Decision Tree, Random Forest and XGBoost all reach 100%. Without the screening answers,
demographics alone predict the outcome poorly:

| Model (no AQ-10 answers or score) | F1, 5-fold CV |
|---|---|
| Logistic Regression | 0.28 ± 0.06 |
| Random Forest | 0.38 ± 0.06 |

For this reason the Streamlit app is an exploration dashboard, not a predictor.

## 🚀 Features

- `assigment clean data.ipynb` — data cleaning walkthrough
- `autism_modeling.ipynb` — cleaning, EDA (Matplotlib, Seaborn, Plotly) and classification models
- `app.py` — Streamlit dashboard:
  - filters by gender, age and who completed the test
  - AQ-10 score distribution and per-question rates by screening outcome
  - screening-outcome rates by gender, jaundice, family history, ethnicity and country
  - the rule-based-label checks and demographic-only model scores, computed live
  - data table with CSV download

## 🛠️ Tech Stack

Python · pandas · scikit-learn · XGBoost · Matplotlib · Seaborn · Plotly · Altair · Streamlit · Jupyter

## 📂 Project Structure

```
Autism-Adult-Data/
├── Autism-Adult-Data.csv          # Dataset (no header row)
├── assigment clean data.ipynb     # Data cleaning notebook
├── autism_modeling.ipynb          # EDA + classification notebook
├── autism_data.py                 # Loading, cleaning and analysis helpers
├── app.py                         # Streamlit dashboard
├── requirements.txt
└── .streamlit/config.toml
```

## 📦 Installation

```bash
git clone https://github.com/amirafarhan/Autism-Adult-Data.git
cd Autism-Adult-Data
python -m venv venv
# Windows: venv\Scripts\activate    Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Run

Streamlit dashboard (opens http://localhost:8501):

```bash
streamlit run app.py
```

The notebooks need a few extra packages:

```bash
pip install jupyter matplotlib seaborn plotly folium xgboost
```

Then run `jupyter notebook`.

## 👩‍💻 Author

Amira Farhan — Machine Learning & Computer Vision Engineer
