import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="뇌졸중 예측 실습실 - 탐색", page_icon="🔍", layout="wide")

st.title("🔍 데이터 탐색")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


df = load_data(DATA_URL)

# -----------------------------
# 1. 나이 / 평균 혈당 히스토그램
# -----------------------------
st.subheader("1️⃣ 나이와 평균 혈당의 분포")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    fig_age = px.histogram(df, x="age", nbins=40, title="나이 분포")
    fig_age.update_layout(xaxis_title="나이", yaxis_title="사람 수")
    st.plotly_chart(fig_age, use_container_width=True)

with hist_col2:
    fig_glucose = px.histogram(df, x="avg_glucose_level", nbins=40, title="평균 혈당 분포")
    fig_glucose.update_layout(xaxis_title="평균 혈당", yaxis_title="사람 수")
    st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()

# -----------------------------
# 2. 뇌졸중 유무별 나이 / 평균 혈당 상자그림 + 평균값 표
# -----------------------------
st.subheader("2️⃣ 뇌졸중 유무에 따른 나이·평균 혈당 비교")

df_group = df.copy()
df_group["뇌졸중 여부"] = df_group["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

box_col1, box_col2 = st.columns(2)

with box_col1:
    fig_box_age = px.box(
        df_group, x="뇌졸중 여부", y="age", color="뇌졸중 여부", title="나이 비교"
    )
    fig_box_age.update_layout(yaxis_title="나이", showlegend=False)
    st.plotly_chart(fig_box_age, use_container_width=True)

with box_col2:
    fig_box_glucose = px.box(
        df_group,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="평균 혈당 비교",
    )
    fig_box_glucose.update_layout(yaxis_title="평균 혈당", showlegend=False)
    st.plotly_chart(fig_box_glucose, use_container_width=True)

mean_table = (
    df_group.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당 평균"})
    .round(2)
)
st.dataframe(mean_table, use_container_width=True)

st.divider()

# -----------------------------
# 3. 고혈압 / 심장병 유무별 뇌졸중 비율 막대그래프
# -----------------------------
st.subheader("3️⃣ 고혈압·심장병 유무에 따른 뇌졸중 비율")

bar_col1, bar_col2 = st.columns(2)

with bar_col1:
    hyper_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hyper_ratio["stroke"] = hyper_ratio["stroke"] * 100
    hyper_ratio["고혈압 여부"] = hyper_ratio["hypertension"].map({0: "없음", 1: "있음"})
    fig_hyper = px.bar(
        hyper_ratio,
        x="고혈압 여부",
        y="stroke",
        title="고혈압 유무별 뇌졸중 비율",
        text_auto=".2f",
    )
    fig_hyper.update_layout(yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig_hyper, use_container_width=True)

with bar_col2:
    heart_ratio = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_ratio["stroke"] = heart_ratio["stroke"] * 100
    heart_ratio["심장병 여부"] = heart_ratio["heart_disease"].map({0: "없음", 1: "있음"})
    fig_heart = px.bar(
        heart_ratio,
        x="심장병 여부",
        y="stroke",
        title="심장병 유무별 뇌졸중 비율",
        text_auto=".2f",
    )
    fig_heart.update_layout(yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# -----------------------------
# 4. bmi 결측자 뇌졸중 비율 vs 전체 뇌졸중 비율
# -----------------------------
st.subheader("4️⃣ 체질량지수(bmi)가 비어 있는 사람들의 뇌졸중 비율")

missing_mask = df["bmi"].isna()
missing_count = int(missing_mask.sum())
missing_ratio = df.loc[missing_mask, "stroke"].mean() * 100 if missing_count > 0 else 0
overall_ratio = df["stroke"].mean() * 100

bmi_compare = pd.DataFrame(
    {
        "구분": ["bmi 결측", "전체"],
        "인원 수": [missing_count, len(df)],
        "뇌졸중 비율(%)": [round(missing_ratio, 2), round(overall_ratio, 2)],
    }
)
st.dataframe(bmi_compare, hide_index=True, use_container_width=True)

st.divider()

# -----------------------------
# 5. 흡연 상태별 인원 수
# -----------------------------
st.subheader("5️⃣ 흡연 상태별 인원 수")

smoking_counts = (
    df["smoking_status"].value_counts().reset_index()
)
smoking_counts.columns = ["흡연 상태", "인원 수"]
st.dataframe(smoking_counts, hide_index=True, use_container_width=True)
