import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="뇌졸중 예측 실습실 - 분류 모델", page_icon="🤖", layout="wide")

st.title("🤖 분류 모델 만들기")
st.caption("stroke = 1은 뇌졸중(양성), stroke = 0은 뇌졸중 아님(음성)이에요.")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


df = load_data(DATA_URL)

# -----------------------------
# 속성(열) 이름 매핑
# -----------------------------
LABEL_TO_COL = {
    "나이": "age",
    "평균 혈당": "avg_glucose_level",
    "체질량지수": "bmi",
    "고혈압": "hypertension",
    "심장병": "heart_disease",
}
COL_TO_LABEL = {v: k for k, v in LABEL_TO_COL.items()}

st.subheader("1️⃣ 입력으로 사용할 속성 고르기")

default_labels = ["나이", "평균 혈당", "고혈압", "심장병"]
selected_labels = st.multiselect(
    "모델의 입력으로 사용할 속성을 골라 주세요.",
    options=list(LABEL_TO_COL.keys()),
    default=default_labels,
)

if len(selected_labels) < 2:
    st.warning("속성을 두 개 이상 골라 주세요.")
    st.stop()

selected_cols = [LABEL_TO_COL[label] for label in selected_labels]

# -----------------------------
# 2. 훈련용 / 테스트용 나누기
# -----------------------------
df_sorted = df.sort_values("id").reset_index(drop=True)
position = np.arange(len(df_sorted))
position_in_group = position % 10
test_mask = position_in_group < 3

train_df = df_sorted.loc[~test_mask].reset_index(drop=True).copy()
test_df = df_sorted.loc[test_mask].reset_index(drop=True).copy()

st.caption(
    f"전체 {len(df_sorted):,}명을 열 명씩 묶어, 각 묶음의 앞 세 명({len(test_df):,}명)을 "
    f"테스트용으로, 나머지 일곱 명({len(train_df):,}명)을 훈련용으로 나누었어요."
)

# bmi를 골랐다면 훈련용 중앙값으로 결측값 채우기
if "bmi" in selected_cols:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.caption(f"체질량지수(bmi)의 빈 값은 훈련용 중앙값인 {bmi_median:.2f}로 채웠어요.")

X_train = train_df[selected_cols].reset_index(drop=True)
y_train = train_df["stroke"].reset_index(drop=True)
X_test = test_df[selected_cols].reset_index(drop=True)
y_test = test_df["stroke"].reset_index(drop=True)

# -----------------------------
# 3. 모델 학습
# -----------------------------
# 로지스틱 회귀는 속성의 크기를 맞춰 주어야 잘 작동해요. (훈련용 기준으로만 크기 맞추기)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_train_acc = accuracy_score(y_train, log_model.predict(X_train_scaled))
log_test_acc = accuracy_score(y_test, log_model.predict(X_test_scaled))

# 의사결정트리는 크기를 맞추지 않은 원래 값을 그대로 사용해요.
tree_model = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
tree_model.fit(X_train, y_train)
tree_train_acc = accuracy_score(y_train, tree_model.predict(X_train))
tree_test_acc = accuracy_score(y_test, tree_model.predict(X_test))

dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_model.fit(X_train, y_train)
dummy_train_acc = accuracy_score(y_train, dummy_model.predict(X_train))
dummy_test_acc = accuracy_score(y_test, dummy_model.predict(X_test))

# -----------------------------
# 4. 정확도 카드
# -----------------------------
st.subheader("2️⃣ 모델별 정확도")

acc_col1, acc_col2, acc_col3 = st.columns(3)

with acc_col1:
    st.metric("로지스틱 회귀(확률로 답하는 모델)", f"{log_test_acc * 100:.1f}%")
    st.caption(f"훈련 정확도 {log_train_acc * 100:.1f}% / 테스트 정확도 {log_test_acc * 100:.1f}%")

with acc_col2:
    st.metric("의사결정트리(질문으로 답하는 모델)", f"{tree_test_acc * 100:.1f}%")
    st.caption(f"훈련 정확도 {tree_train_acc * 100:.1f}% / 테스트 정확도 {tree_test_acc * 100:.1f}%")

with acc_col3:
    st.metric("다수결 모델(입력을 보지 않고 많은 쪽으로 답하는 모델)", f"{dummy_test_acc * 100:.1f}%")
    st.caption(f"훈련 정확도 {dummy_train_acc * 100:.1f}% / 테스트 정확도 {dummy_test_acc * 100:.1f}%")

st.divider()

# -----------------------------
# 5. 산점도 + 결정 경계 + 결정트리 영역
# -----------------------------
st.subheader("3️⃣ 두 속성으로 그려 보는 그림")

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_label = st.selectbox("가로축으로 사용할 속성", selected_labels, index=0)
with axis_col2:
    y_options = [label for label in selected_labels if label != x_label]
    y_label = st.selectbox("세로축으로 사용할 속성", y_options, index=0)

x_col = LABEL_TO_COL[x_label]
y_col = LABEL_TO_COL[y_label]
other_cols = [c for c in selected_cols if c not in (x_col, y_col)]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정
fixed_values = {c: test_df[c].median() for c in other_cols}
if fixed_values:
    fixed_text = ", ".join(
        f"{COL_TO_LABEL[c]} = {v:.2f}" for c, v in fixed_values.items()
    )
    st.caption(f"그림의 두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정했어요. ({fixed_text})")

x_min, x_max = float(test_df[x_col].min()), float(test_df[x_col].max())
y_min, y_max = float(test_df[y_col].min()), float(test_df[y_col].max())
x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
x_range = (x_min - x_pad, x_max + x_pad)
y_range = (y_min - y_pad, y_max + y_pad)

fig = go.Figure()

# --- 의사결정트리가 나눈 칸을 옅은 색으로 칠하기 ---
grid_n = 120
xs = np.linspace(x_range[0], x_range[1], grid_n)
ys = np.linspace(y_range[0], y_range[1], grid_n)
xx, yy = np.meshgrid(xs, ys)

grid_df = pd.DataFrame({c: np.full(xx.size, fixed_values.get(c, 0.0)) for c in selected_cols})
grid_df[x_col] = xx.ravel()
grid_df[y_col] = yy.ravel()
grid_df = grid_df[selected_cols]

zz = tree_model.predict(grid_df).reshape(xx.shape)

fig.add_trace(
    go.Heatmap(
        x=xs,
        y=ys,
        z=zz,
        zmin=0,
        zmax=1,
        colorscale=[[0, "#BBDEFB"], [1, "#FFCDD2"]],
        showscale=False,
        opacity=0.35,
        hoverinfo="skip",
    )
)

# --- 테스트 데이터 산점도 (실제 뇌졸중 여부로 색 구분) ---
for label_value, kor_name, color in [(0, "뇌졸중 아님", "#1E88E5"), (1, "뇌졸중", "#E53935")]:
    subset = test_df[y_test == label_value]
    fig.add_trace(
        go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode="markers",
            name=kor_name,
            marker=dict(color=color, size=7, opacity=0.75),
        )
    )

# --- 로지스틱 회귀의 0.5 결정 경계선 ---
feature_index = {c: i for i, c in enumerate(selected_cols)}
coef = log_model.coef_[0]
intercept = log_model.intercept_[0]
mean_ = scaler.mean_
scale_ = scaler.scale_

# 원래 값 기준 계수: a_i = coef_i / scale_i
a = coef / scale_
# 상수항: intercept - sum(a_i * mean_i) 를 이항 정리
const_term = intercept - np.sum(a * mean_)

a_x = a[feature_index[x_col]]
a_y = a[feature_index[y_col]]

fixed_contribution = sum(a[feature_index[c]] * fixed_values[c] for c in other_cols)

boundary_drawn = False
if abs(a_y) > 1e-12:
    x_line = np.linspace(x_range[0], x_range[1], 200)
    # a_x*x + a_y*y + fixed_contribution + const_term = 0
    y_line = -(a_x * x_line + fixed_contribution + const_term) / a_y

    within_range = (y_line >= y_range[0]) & (y_line <= y_range[1])
    if within_range.any():
        fig.add_trace(
            go.Scatter(
                x=x_line[within_range],
                y=y_line[within_range],
                mode="lines",
                name="로지스틱 회귀 결정 경계(0.5)",
                line=dict(color="black", width=2, dash="dash"),
            )
        )
        boundary_drawn = True
        if not within_range.all():
            st.caption("로지스틱 회귀의 결정 경계선 가운데 일부는 그림 범위를 벗어나 보이지 않아요.")
    else:
        st.caption("로지스틱 회귀의 결정 경계선이 이 그림의 범위 밖에 있어서 보이지 않아요.")
elif abs(a_x) > 1e-12:
    x_line_value = -(fixed_contribution + const_term) / a_x
    if x_range[0] <= x_line_value <= x_range[1]:
        fig.add_trace(
            go.Scatter(
                x=[x_line_value, x_line_value],
                y=list(y_range),
                mode="lines",
                name="로지스틱 회귀 결정 경계(0.5)",
                line=dict(color="black", width=2, dash="dash"),
            )
        )
        boundary_drawn = True
    else:
        st.caption("로지스틱 회귀의 결정 경계선이 이 그림의 범위 밖에 있어서 보이지 않아요.")
else:
    st.caption("이 축들로는 로지스틱 회귀의 결정 경계선을 그릴 수 없어요.")

fig.update_layout(
    title=f"{x_label} vs {y_label} (배경: 의사결정트리가 나눈 칸)",
    xaxis_title=x_label,
    yaxis_title=y_label,
    xaxis=dict(range=list(x_range)),
    yaxis=dict(range=list(y_range)),
    legend_title="구분",
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# 6. 의사결정트리 가지 그림
# -----------------------------
st.subheader("4️⃣ 의사결정트리가 던진 질문")

tree_ = tree_model.tree_
classes_ = tree_model.classes_
feature_names_kor = [COL_TO_LABEL[c] for c in selected_cols]

leaf_info = []
used_feature_idx = set()


def build_dot(node_id: int) -> list:
    lines = []
    n_samples = int(tree_.n_node_samples[node_id])
    value = tree_.value[node_id][0]
    if 1 in classes_:
        idx1 = int(np.where(classes_ == 1)[0][0])
        stroke_count = int(round(value[idx1]))
    else:
        stroke_count = 0
    ratio = (stroke_count / n_samples * 100) if n_samples > 0 else 0.0

    is_leaf = tree_.children_left[node_id] == -1

    if is_leaf:
        pred_class = classes_[int(np.argmax(value))]
        pred_label = "뇌졸중" if pred_class == 1 else "아님"
        fill_color = "#FFCDD2" if pred_class == 1 else "#BBDEFB"
        label = (
            f"인원 {n_samples}명\\n뇌졸중 {stroke_count}명 ({ratio:.1f}%)\\n예측: {pred_label}"
        )
        lines.append(f'{node_id} [label="{label}", fillcolor="{fill_color}"];')
        leaf_info.append(pred_class)
    else:
        feat_idx = tree_.feature[node_id]
        used_feature_idx.add(feat_idx)
        feat_name = feature_names_kor[feat_idx]
        threshold = tree_.threshold[node_id]
        label = (
            f"{feat_name} ≤ {threshold:.2f} ?\\n인원 {n_samples}명\\n"
            f"뇌졸중 {stroke_count}명 ({ratio:.1f}%)"
        )
        lines.append(f'{node_id} [label="{label}", fillcolor="#FFFFFF"];')

        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        lines.extend(build_dot(left))
        lines.extend(build_dot(right))
        lines.append(f'{node_id} -> {left} [label="예"];')
        lines.append(f'{node_id} -> {right} [label="아니요"];')

    return lines


dot_body = build_dot(0)
dot_string = (
    "digraph Tree {\n"
    'node [shape=box, style="filled, rounded"];\n'
    + "\n".join(dot_body)
    + "\n}"
)

st.graphviz_chart(dot_string)

leaf_total = len(leaf_info)
no_stroke_leaf_total = sum(1 for pred in leaf_info if pred == 0)

st.write(f"- 답을 내는 마디(잎)는 모두 **{leaf_total}칸**이고, 그중 **{no_stroke_leaf_total}칸**이 '아님'이라고 답해요.")

st.write("- 고른 속성 가운데 이 나무가 실제로 물어본 것:")
for col in selected_cols:
    idx = selected_cols.index(col)
    used = "물어봤어요" if idx in used_feature_idx else "물어보지 않았어요"
    st.write(f"  - {COL_TO_LABEL[col]}: {used}")
