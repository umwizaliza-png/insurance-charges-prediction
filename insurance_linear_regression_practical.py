
"""
INSURANCE CHARGES PREDICTION
Professional Machine Learning Dashboard

"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insurance Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# PROFESSIONAL COLORFUL DASHBOARD THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #101827;
    color: #F1F5F9;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #60A5FA !important;
    font-weight: 800 !important;
}

h2 {
    color: #38BDF8 !important;
    font-weight: 700 !important;
}

h3 {
    color: #2DD4BF !important;
    font-weight: 600 !important;
}

p, label, li {
    color: #CBD5E1;
}

[data-testid="stMetric"] {
    background-color: #1C2940;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px;
}

[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
}

[data-testid="stMetricValue"] {
    color: #60A5FA !important;
    font-weight: 800;
}

[data-testid="stMetricDelta"] {
    color: #2DD4BF !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 12px;
}

[data-testid="stExpander"] {
    background-color: #1C2940;
    border: 1px solid #334155;
    border-radius: 12px;
}

.stAlert {
    border-radius: 12px;
}

hr {
    border-color: #334155;
}

.stCaption {
    color: #94A3B8;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.title("🏥 Insurance Analytics Dashboard")

st.markdown("""
Analyze medical insurance data, explore important features,
evaluate a Linear Regression model, and estimate insurance charges.
""")

st.markdown("""
<div style="
    background-color:#1C2940;
    padding:15px;
    border-radius:12px;
    border-left:5px solid #3B82F6;
">
    <span style="color:#60A5FA;font-weight:bold;">
        MACHINE LEARNING PRACTICAL
    </span>
    <br>
    <span style="color:#CBD5E1;">
        Algorithm: Linear Regression | Dataset: Medical Insurance
        | Questions 6–13
    </span>
</div>
""", unsafe_allow_html=True)

st.write("")


# ============================================================
# DATASET
# ============================================================

DATA_PATH = Path("insurance.csv")

if not DATA_PATH.exists():
    st.error(
        "insurance.csv was not found. "
        "Place it beside this Python file in your GitHub repository."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

required_columns = [
    "age", "sex", "bmi", "children",
    "smoker", "region", "charges"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

df = df[required_columns].dropna().copy()


# ============================================================
# LOAD AND DISPLAY DATASET
# ============================================================

st.divider()

st.header("📊  Dataset Exploration")

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Total Records", f"{len(df):,}")
metric2.metric("Total Features", "6")
metric3.metric("Target Variable", "Charges")

with st.expander("View First Five Rows", expanded=True):
    st.dataframe(
        df.head(),
        use_container_width=True,
        hide_index=True
    )

with st.expander("View Summary Statistics"):
    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ============================================================
# ONE-HOT ENCODING
# ============================================================

st.divider()

st.header("🔠  Categorical Variables")

st.markdown("""
Categorical features are converted into numerical features
using One-Hot Encoding.
""")

df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

with st.expander("View Encoded Dataset", expanded=True):
    st.dataframe(
        df_encoded.head(),
        use_container_width=True,
        hide_index=True
    )

    st.caption("Encoded feature names:")
    st.write(df_encoded.columns.tolist())


# ============================================================
# TRAINING AND TESTING SPLIT
# ============================================================

st.divider()

st.header("📂 Training and Testing Split")

X = df_encoded.drop(columns=["charges"])
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Dataset · 80%",
        f"{len(X_train):,} records"
    )

with col2:
    st.metric(
        "Testing Dataset · 20%",
        f"{len(X_test):,} records"
    )


# ============================================================
# LINEAR REGRESSION MODEL
# ============================================================

st.divider()

st.header("🤖  Model Training")

model = LinearRegression()
model.fit(X_train, y_train)

st.success("Linear Regression model trained successfully.")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

with st.expander("View Model Coefficients", expanded=True):
    st.dataframe(
        coefficients,
        use_container_width=True,
        hide_index=True
    )

st.markdown("### Model Intercept")

st.code(f"{model.intercept_:.4f}")


# ============================================================
# REGRESSION EQUATION
# ============================================================

st.divider()

st.header("🧮  Regression Equation")

equation = f"charges = {model.intercept_:.4f}"

for feature, coefficient in zip(X.columns, model.coef_):
    sign = "+" if coefficient >= 0 else "-"
    equation += (
        f" {sign} {abs(coefficient):.4f}({feature})"
    )

with st.expander("View Complete Equation", expanded=True):
    st.code(equation, language="text")


# ============================================================
# MODEL EVALUATION
# ============================================================

st.divider()

st.header("📈  Model Evaluation")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"${mae:,.2f}"
    )

with col2:
    st.metric(
        "MSE",
        f"${mse:,.2f}"
    )

with col3:
    st.metric(
        "R² Score",
        f"{r2 * 100:.2f}%"
    )

st.caption(
    "R² represents the proportion of variation explained by the model. "
    "It is not classification accuracy."
)


# ============================================================
# ACTUAL VS PREDICTED REGRESSION PLOT
# ============================================================

st.divider()

st.header("📉  Regression Plot")

fig1, ax1 = plt.subplots(figsize=(10, 6))

fig1.patch.set_facecolor("#101827")
ax1.set_facecolor("#1C2940")

ax1.scatter(
    y_test,
    y_pred,
    color="#38BDF8",
    alpha=0.75,
    s=45,
    label="Predicted charges"
)

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())

ax1.plot(
    [line_min, line_max],
    [line_min, line_max],
    color="#F87171",
    linestyle="--",
    linewidth=2.5,
    label="Ideal prediction"
)

ax1.set_xlabel(
    "Actual Insurance Charges ($)",
    color="#F1F5F9"
)

ax1.set_ylabel(
    "Predicted Insurance Charges ($)",
    color="#F1F5F9"
)

ax1.set_title(
    "Actual vs Predicted Insurance Charges",
    color="#60A5FA",
    fontsize=14,
    fontweight="bold"
)

ax1.tick_params(colors="#CBD5E1")

for spine in ax1.spines.values():
    spine.set_color("#334155")

ax1.grid(color="#334155", alpha=0.5)

ax1.legend(
    facecolor="#1C2940",
    edgecolor="#334155",
    labelcolor="#F1F5F9"
)

fig1.tight_layout()

st.pyplot(fig1)

plt.close(fig1)


# ============================================================
# CONFUSION MATRIX EXPLANATION
# ============================================================

st.markdown("###: Confusion Matrix Heatmap")

st.info("""
A confusion matrix is suitable for classification models.
Linear Regression predicts continuous insurance charges.
Therefore, a confusion matrix is not directly appropriate
for this regression task.
""")


# ============================================================
# PREDICT CHARGES
# ============================================================

st.divider()

st.header("💰 Insurance Charges Prediction")

st.markdown("""
### Individual Information
""")

new_person = pd.DataFrame([{
    "age": 19,
    "sex": "female",
    "bmi": 27.9,
    "children": 0,
    "smoker": "yes",
    "region": "southwest"
}])

st.dataframe(
    new_person,
    use_container_width=True,
    hide_index=True
)

new_person_encoded = pd.get_dummies(
    new_person,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

new_person_encoded = new_person_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

predicted_charge = model.predict(
    new_person_encoded
)[0]

st.markdown("""
<div style="
    background-color:#1C2940;
    border:1px solid #14B8A6;
    border-left:6px solid #14B8A6;
    border-radius:14px;
    padding:22px;
">
    <div style="
        color:#2DD4BF;
        font-size:16px;
        font-weight:bold;
    ">
        ESTIMATED INSURANCE CHARGES
    </div>
    <div style="
        color:#F8FAFC;
        font-size:36px;
        font-weight:800;
        margin-top:8px;
    ">
        $""" + f"{predicted_charge:,.2f}" + """
    </div>
    <div style="color:#94A3B8;font-size:13px;">
        Linear Regression Prediction
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#64748B;
    font-size:13px;
    padding:10px;
">
    INSURANCE ANALYTICS DASHBOARD
    <br>
    Machine Learning Practical · Linear Regression
</div>
""", unsafe_allow_html=True)
