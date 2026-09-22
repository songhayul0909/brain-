import streamlit as st
import pandas as pd

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 뇌졸중 예측 실습실")

# -----------------------------
# 데이터 불러오기
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


df = load_data(DATA_URL)

# -----------------------------
# 큰 숫자 카드 네 개
# -----------------------------
total_count = len(df)
col_count = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total_count * 100

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_count:,}명")
card2.metric("열 개수", f"{col_count}개")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
card4.metric("stroke 비율", f"{stroke_ratio:.2f}%")

st.divider()

# -----------------------------
# 열 이름 · 우리말 뜻 · 값의 종류 · 빈 값 개수 표
# -----------------------------
st.subheader("📋 열(컬럼) 설명표")
st.caption("‘우리말 뜻’ 칸은 비어 있어요. 교재를 보고 직접 채워 보세요.")


def get_value_kind(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "숫자(정수)"
    elif pd.api.types.is_float_dtype(series):
        return "숫자(실수)"
    else:
        return "문자(범주형)"


column_info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": ["" for _ in df.columns],
        "값의 종류": [get_value_kind(df[col]) for col in df.columns],
        "빈 값 개수": [df[col].isna().sum() for col in df.columns],
    }
)

edited_info = st.data_editor(
    column_info,
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    key="column_info_editor",
)

st.divider()

# -----------------------------
# 데이터 처음 다섯 줄
# -----------------------------
st.subheader("🔎 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# -----------------------------
# 데이터 출처
# -----------------------------
st.subheader("📚 데이터 출처")
st.text_area(
    "교재에 나온 데이터 출처를 여기에 적어 보세요.",
    placeholder="여기에 데이터 출처를 입력하세요.",
    height=120,
)
