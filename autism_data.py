"""Loading, cleaning and analysis helpers for the UCI Autism Screening Adult dataset.

Cleaning follows the notebooks: '?' -> missing, age to numeric, the age outlier
(>= 80, a data-entry error of 383) removed, country/ethnicity lower-cased and
stripped of quotes. Column names keep the dataset's original spelling
(``ethicity``, ``jundice``, ``austim``, ``realation``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "Autism-Adult-Data.csv"
ITEMS = [f"A{i}_Score" for i in range(1, 11)]
COLUMNS = ITEMS + ["age", "gender", "ethicity", "jundice", "austim", "country_of_res",
                   "used_app_before", "result", "age_desc", "realation", "class/ASD"]
THRESHOLD = 7  # AQ-10 screening cut-off: a score of 7 or more is flagged
LABELS = {
    "gender": "Gender", "ethicity": "Ethnicity", "jundice": "Born with jaundice",
    "austim": "Family member with autism", "country_of_res": "Country of residence",
    "realation": "Who completed the test", "result": "AQ-10 score", "class/ASD": "Screening outcome",
    "age": "Age",
}


def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, names=COLUMNS, na_values="?")


def load_clean(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = load_raw(path)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[~(df["age"] >= 80)].copy()  # keep missing ages, drop the 383 outlier
    for col in ["country_of_res", "ethicity", "realation"]:
        # object dtype keeps missing values as NaN (pd.NA breaks scikit-learn imputers)
        df[col] = df[col].str.lower().str.replace("'", "", regex=False).str.strip()
    df["gender"] = df["gender"].map({"m": "male", "f": "female"})
    df["class/ASD"] = df["class/ASD"].str.upper()
    return df.reset_index(drop=True)


def leakage_checks(df: pd.DataFrame) -> dict:
    """Verify how the label relates to the screening answers."""
    item_sum = df[ITEMS].sum(axis=1)
    flagged = df["result"] >= THRESHOLD
    return {
        "rows": len(df),
        "result_is_item_sum": float((item_sum == df["result"]).mean()),
        "label_is_threshold": float((flagged == (df["class/ASD"] == "YES")).mean()),
    }


def demographics_only_f1(df: pd.DataFrame, seed: int = 42) -> dict:
    """5-fold CV F1 of models that never see the AQ-10 answers or score."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat = ["gender", "ethicity", "jundice", "austim", "country_of_res", "realation"]
    X, y = df[["age"] + cat], (df["class/ASD"] == "YES").astype(int)
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age"]),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = {}
    for name, model in [("Logistic Regression", LogisticRegression(max_iter=2000)),
                        ("Random Forest", RandomForestClassifier(random_state=seed))]:
        scores = cross_val_score(make_pipeline(pre, model), X, y, cv=cv, scoring="f1")
        out[name] = (float(scores.mean()), float(scores.std()))
    return out
