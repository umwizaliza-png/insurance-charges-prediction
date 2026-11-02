
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
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# CUSTOM DASHBOARD THEME
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

[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 12px;
}

.stAlert {
    border-radius: 12px;
}

hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.title("🏥 Insurance Charges Prediction")

st.markdown("""
Welcome to the Insurance Charges Prediction application.

This application uses Machine Learning and Linear Regression
to estimate individual medical insurance charges based on
personal and lifestyle information.
""")

st.markdown("""
<div style="
    background-color:#1C2940;
    padding:18px;
    border-radius:12px;
    border-left:5px solid #3B82F6;
">
    <span style="color:#60A5FA;font-weight:bold;">
        MACHINE LEARNING APPLICATION
    </span>
    <br>
    <span style="color:#CBD5E1;">
        Algorithm: Linear Regression |
        Dataset: Medical Insurance |
        Prediction: Insurance Charges (USD)
    </span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = Path(__file__).parent / "insurance.csv"

if not DATA_PATH.exists():
    st.error("The insurance.csv dataset was not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)

required_columns = [
    "age", "sex", "bmi", "children",
    "smoker", "region", "charges"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

df = df[required_columns].dropna().copy()

if df.empty:
    st.error("The dataset contains no usable records.")
    st.stop()


# ============================================================
# DATASET EXPLORATION
# ============================================================

st.divider()
st.header("📊 Dataset Exploration")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{len(df):,}")
col2.metric("Total Features", "6")
col3.metric("Target Variable", "Charges")

with st.expander("View Dataset", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("View Summary Statistics"):
    st.dataframe(df.describe(), use_container_width=True)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

st.divider()
st.header("🔠 Categorical Variables")

st.write(
    "Categorical variables are converted into numerical "
    "features using One-Hot Encoding."
)

df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)


# ============================================================
# TRAINING AND TESTING SPLIT
# ============================================================

X = df_encoded.drop(columns=["charges"])
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

st.divider()
st.header("📂 Training and Testing")

col1, col2 = st.columns(2)

col1.metric("Training Records (80%)", f"{len(X_train):,}")
col2.metric("Testing Records (20%)", f"{len(X_test):,}")


# ============================================================
# LINEAR REGRESSION MODEL
# ============================================================

st.divider()
st.header("🤖 Model Training")

model = LinearRegression()
model.fit(X_train, y_train)

st.success("Linear Regression model trained successfully.")

y_pred = model.predict(X_test)


# ============================================================
# MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Mean Absolute Error", f"${mae:,.2f}")
col2.metric("Mean Squared Error", f"${mse:,.2f}")
col3.metric("R² Score", f"{r2 * 100:.2f}%")

st.caption(
    "R² represents the proportion of variation in insurance "
    "charges explained by the model. It is not classification accuracy."
)


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

with st.expander("View Model Coefficients"):
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    st.dataframe(coefficients, use_container_width=True)

    st.write("Model Intercept")
    st.code(f"{model.intercept_:.4f}")


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

st.divider()
st.header("📉 Actual vs Predicted Charges")

fig, ax = plt.subplots(figsize=(10, 6))

fig.patch.set_facecolor("#101827")
ax.set_facecolor("#1C2940")

ax.scatter(
    y_test,
    y_pred,
    color="#38BDF8",
    alpha=0.75,
    s=45
)

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())

ax.plot(
    [line_min, line_max],
    [line_min, line_max],
    color="#F87171",
    linestyle="--",
    linewidth=2
)

ax.set_xlabel("Actual Charges ($)", color="#F1F5F9")
ax.set_ylabel("Predicted Charges ($)", color="#F1F5F9")
ax.set_title(
    "Actual vs Predicted Insurance Charges",
    color="#60A5FA"
)

ax.tick_params(colors="#CBD5E1")
ax.grid(color="#334155", alpha=0.5)

for spine in ax.spines.values():
    spine.set_color("#334155")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ============================================================
# INDIVIDUAL INSURANCE CHARGES PREDICTION
# ============================================================

st.divider()
st.header("💰 Individual Insurance Charges Prediction")

st.write(
    "Enter the individual's information below to estimate "
    "their medical insurance charges."
)

with st.form("insurance_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

        sex = st.selectbox(
            "Sex",
            ["female", "male"]
        )

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=70.0,
            value=27.9,
            step=0.1
        )

    with col2:
        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        smoker = st.selectbox(
            "Smoking Status",
            ["no", "yes"]
        )

        region = st.selectbox(
            "Residential Region",
            [
                "southwest",
                "southeast",
                "northwest",
                "northeast"
            ]
        )

    submitted = st.form_submit_button(
        "Predict Insurance Charges",
        use_container_width=True
    )


# ============================================================
# GENERATE PREDICTION
# ============================================================

if submitted:

    new_person = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # Apply the same One-Hot Encoding methodology
    new_person_encoded = pd.get_dummies(
        new_person,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int
    )

    # Match the exact training feature columns
    new_person_encoded = new_person_encoded.reindex(
        columns=X.columns,
        fill_value=0
    )

    predicted_charge = model.predict(
        new_person_encoded
    )[0]

    st.divider()
    st.subheader("🎯 Prediction Result")

    st.markdown(f"""
    <div style="
        background-color:#1C2940;
        border:1px solid #14B8A6;
        border-left:6px solid #14B8A6;
        border-radius:14px;
        padding:25px;
    ">
        <div style="
            color:#2DD4BF;
            font-size:17px;
            font-weight:bold;
        ">
            ESTIMATED INSURANCE CHARGES
        </div>
        <div style="
            color:#F8FAFC;
            font-size:38px;
            font-weight:800;
            margin-top:10px;
        ">
            ${predicted_charge:,.2f}
        </div>
        <div style="color:#94A3B8;font-size:13px;">
            Predicted using Linear Regression
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Individual Information")

    st.dataframe(
        new_person,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This result is a statistical estimate based on the "
        "training dataset. It is not an actual insurance quote."
    )


# ============================================================
# DISCLAIMER AND FOOTER
# ============================================================

st.divider()

st.warning(
    "Academic Machine Learning Prototype: Predictions are "
    "estimates based on the available dataset. They should "
    "not be treated as official insurance prices or medical advice."
)

st.markdown("""
<div style="
    text-align:center;
    color:#64748B;
    font-size:13px;
    padding:15px;
">
    INSURANCE CHARGES PREDICTION APPLICATION
    <br>
    Machine Learning Practical | Linear Regression
</div>
""", unsafe_allow_html=True)
