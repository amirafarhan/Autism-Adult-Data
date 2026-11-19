"""Streamlit EDA dashboard for the Autism Screening Adult dataset.

Run with:  streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

import autism_data as ad

st.set_page_config(page_title="Autism Screening Data Explorer", page_icon="🧩", layout="wide")
COLORS = alt.Scale(domain=["NO", "YES"], range=["#8d99ae", "#e76f51"])


@st.cache_data
def get_data() -> pd.DataFrame:
    return ad.load_clean()


@st.cache_data
def get_checks(df: pd.DataFrame) -> dict:
    return ad.leakage_checks(ad.load_raw())


@st.cache_data
def get_demo_f1(df: pd.DataFrame) -> dict:
    return ad.demographics_only_f1(df)


try:
    df_all = get_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

st.title("🧩 Autism Screening Data Explorer")
st.caption("Exploratory analysis of the UCI *Autism Screening Adult* dataset (AQ-10 questionnaire, 704 adults).")
st.info("**Not a diagnostic tool.** This dashboard explores survey data. The AQ-10 is a short screening questionnaire; "
        "only a qualified clinician can assess autism.", icon="ℹ️")

# ---------------- Sidebar filters ----------------
with st.sidebar:
    st.header("Filters")
    genders = st.multiselect("Gender", sorted(df_all["gender"].dropna().unique()), placeholder="All")
    ages = df_all["age"].dropna()
    age_range = st.slider("Age", int(ages.min()), int(ages.max()), (int(ages.min()), int(ages.max())))
    include_missing_age = st.checkbox("Include rows with missing age", value=True)
    relation = st.multiselect("Who completed the test", sorted(df_all["realation"].dropna().unique()), placeholder="All")

df = df_all
if genders:
    df = df[df["gender"].isin(genders)]
if relation:
    df = df[df["realation"].isin(relation)]
in_range = df["age"].between(*age_range)
df = df[in_range | (df["age"].isna() & include_missing_age)]

if df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Respondents", len(df))
c2.metric("Flagged by screening (score ≥ 7)", f"{(df['class/ASD'] == 'YES').mean():.1%}")
c3.metric("Median age", f"{df['age'].median():.0f}" if df["age"].notna().any() else "–")
c4.metric("Mean AQ-10 score", f"{df['result'].mean():.1f} / 10")

tab_scores, tab_demo, tab_label, tab_data = st.tabs(
    ["📊 Screening scores", "👥 Demographics", "🔍 Why the label is rule-based", "🗂️ Data"])

# ---------------- Scores ----------------
with tab_scores:
    left, right = st.columns(2)
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X("result:O", title="AQ-10 score (0–10)"),
        y=alt.Y("count()", title="Respondents"),
        color=alt.Color("class/ASD:N", scale=COLORS, title="Screening outcome"),
        tooltip=["result", "class/ASD", "count()"],
    )
    left.altair_chart(hist.properties(title="Score distribution — every score ≥ 7 is labelled YES", height=320),
                      width="stretch")

    items = (df.groupby("class/ASD")[ad.ITEMS].mean().T.reset_index(names="item")
             .melt("item", var_name="class/ASD", value_name="rate"))
    items["item"] = items["item"].str.replace("_Score", "")
    right.altair_chart(
        alt.Chart(items).mark_bar().encode(
            x=alt.X("item:N", sort=[f"A{i}" for i in range(1, 11)], title="Question"),
            xOffset="class/ASD:N",
            y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title="Share scoring 1"),
            color=alt.Color("class/ASD:N", scale=COLORS, title="Screening outcome"),
            tooltip=["item", "class/ASD", alt.Tooltip("rate:Q", format=".1%")],
        ).properties(title="How often each question scores a point, by outcome", height=320),
        width="stretch",
    )
    st.caption("The dataset only contains item scores (0/1), not the question wording.")

# ---------------- Demographics ----------------
def rate_chart(frame: pd.DataFrame, col: str, top: int | None = None, title: str = ""):
    g = frame.dropna(subset=[col]).groupby(col).agg(n=("class/ASD", "size"), rate=("class/ASD", lambda s: (s == "YES").mean()))
    g = g.sort_values("n", ascending=False)
    if top:
        g = g.head(top)
    g = g.reset_index()
    return alt.Chart(g).mark_bar(color="#2a9d8f").encode(
        y=alt.Y(f"{col}:N", sort="-x", title=None),
        x=alt.X("rate:Q", axis=alt.Axis(format="%"), title="Flagged by screening", scale=alt.Scale(domain=[0, 1])),
        tooltip=[col, alt.Tooltip("rate:Q", format=".1%", title="flagged"), alt.Tooltip("n:Q", title="respondents")],
    ).properties(title=title or ad.LABELS[col], height=alt.Step(22))


with tab_demo:
    a, b = st.columns(2)
    a.altair_chart(
        alt.Chart(df.dropna(subset=["age"])).mark_bar(opacity=0.85).encode(
            x=alt.X("age:Q", bin=alt.Bin(step=5), title="Age"),
            y=alt.Y("count()", stack=True, title="Respondents"),
            color=alt.Color("class/ASD:N", scale=COLORS, title="Screening outcome"),
        ).properties(title="Age distribution", height=300), width="stretch")
    b.altair_chart(rate_chart(df, "gender"), width="stretch")
    a.altair_chart(rate_chart(df, "jundice"), width="stretch")
    b.altair_chart(rate_chart(df, "austim"), width="stretch")
    a.altair_chart(rate_chart(df, "ethicity", title="Ethnicity"), width="stretch")
    b.altair_chart(rate_chart(df, "country_of_res", top=12, title="Country of residence (12 most common)"), width="stretch")
    st.caption("Groups with few respondents give unstable rates — hover a bar to see its size. "
               f"{int(df_all['ethicity'].isna().sum())} rows have no ethnicity and are omitted from that chart.")

# ---------------- Label analysis ----------------
with tab_label:
    checks = get_checks(df_all)
    st.subheader("The screening outcome is a fixed rule, not something to learn")
    k1, k2 = st.columns(2)
    k1.metric("Rows where score = sum of A1–A10", f"{checks['result_is_item_sum']:.0%}", f"{checks['rows']} rows checked",
              delta_color="off")
    k2.metric("Rows where label = (score ≥ 7)", f"{checks['label_is_threshold']:.0%}", f"{checks['rows']} rows checked",
              delta_color="off")
    st.markdown(
        "Because the label is defined by the score, any model given the score or the ten answers simply rediscovers the "
        "cut-off — which is why the notebook's Logistic Regression, Decision Tree, Random Forest and XGBoost all reach 100%. "
        "A meaningful question is how much the **other** information predicts the outcome:")
    with st.spinner("Cross-validating demographic-only models..."):
        f1 = get_demo_f1(df_all)
    st.dataframe(pd.DataFrame([{"Model (no AQ-10 answers or score)": k, "F1 (5-fold CV)": f"{m:.3f} ± {s:.3f}"}
                               for k, (m, s) in f1.items()]), hide_index=True, width="stretch")
    st.caption("Demographics alone predict the screening outcome poorly, so no prediction feature is offered here.")

# ---------------- Data ----------------
with tab_data:
    st.dataframe(df.rename(columns=ad.LABELS), hide_index=True, width="stretch")
    st.download_button("Download filtered data (CSV)", df.to_csv(index=False), "autism_filtered.csv", "text/csv")
