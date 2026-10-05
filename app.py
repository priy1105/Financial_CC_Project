"""Interactive dashboard for the credit card fraud detection project."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "card_transdata.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURES = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
    "home_to_last_ratio",
    "transaction_score",
    "price_category",
]

# These are the saved test-set results printed by the notebook (20% holdout).
RESULTS = {
    "Logistic Regression": {
        "accuracy": 0.916475,
        "precision": 0.51,
        "recall": 0.94,
        "f1": 0.66,
        "matrix": [[166976, 15581], [1124, 16319]],
    },
    "Random Forest": {
        "accuracy": 0.942715,
        "precision": 0.60,
        "recall": 0.99,
        "f1": 0.75,
        "matrix": [[171253, 11304], [153, 17290]],
    },
}


@st.cache_data(show_spinner="Loading the transaction dataset…")
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def render_image(filename, caption):
    path = BASE_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Chart image not found: {filename}")


st.set_page_config(
    page_title="Credit Card Fraud | Project Showcase",
    page_icon="💳",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: #f4f7fb; padding: 1rem; border-radius: 0.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💳 Credit Card Fraud Detection")
st.caption("Machine learning project showcase · 1 million transaction records")

page = st.sidebar.radio(
    "Explore the project",
    ["Project overview", "Data exploration", "Model evaluation", "Try a transaction"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Models: Logistic Regression · Random Forest")

if page == "Project overview":
    st.subheader("Project at a glance")
    st.write(
        "This project compares two classifiers on an imbalanced transaction dataset. "
        "The main evaluation focus is fraud recall, while keeping false alerts visible."
    )

    if DATA_PATH.exists():
        try:
            data = load_dataset()
            fraud_count = int(data["fraud"].sum())
            total_count = len(data)
            legit_count = total_count - fraud_count
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transactions", f"{total_count:,}")
            c2.metric("Fraud cases", f"{fraud_count:,}", f"{fraud_count / total_count:.2%} of data")
            c3.metric("Legitimate", f"{legit_count:,}")
            c4.metric("Best fraud recall", "99%", "Random Forest test set")
            st.bar_chart(
                pd.DataFrame(
                    {"Transactions": [legit_count, fraud_count]},
                    index=["Legitimate", "Fraud"],
                ),
                color="#3969ac",
            )
        except Exception as exc:
            st.warning(f"Could not read the dataset: {exc}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset size", "1,000,000")
        c2.metric("Random Forest accuracy", "94.27%")
        c3.metric("Fraud recall", "99%")
        st.info("Add card_transdata.csv beside app.py to show live dataset counts.")

    st.markdown("#### Notebook visuals")
    left, right = st.columns(2)
    with left:
        render_image("class distribution.png", "Fraud and legitimate transaction distribution")
    with right:
        render_image("corr heat.png", "Feature correlation heatmap")

elif page == "Data exploration":
    st.subheader("Explore the transaction data")
    if not DATA_PATH.exists():
        st.error("card_transdata.csv was not found beside app.py.")
    else:
        try:
            data = load_dataset()
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(data):,}")
            c2.metric("Columns", str(data.shape[1]))
            c3.metric("Missing values", f"{int(data.isna().sum().sum()):,}")

            st.markdown("#### Class balance")
            counts = data["fraud"].value_counts().reindex([0, 1], fill_value=0)
            st.bar_chart(
                pd.DataFrame(
                    {"Transactions": [int(counts.loc[0]), int(counts.loc[1])]},
                    index=["Legitimate", "Fraud"],
                ),
                color="#3969ac",
            )

            st.markdown("#### Correlations")
            render_image("corr heat.png", "Correlation heatmap from the notebook")
            st.markdown("#### Numeric feature distributions and outliers")
            render_image("fraud outliers.png", "Outlier boxplots from the notebook")
            with st.expander("View descriptive statistics"):
                st.dataframe(data.describe().T, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not analyze the dataset: {exc}")

elif page == "Model evaluation":
    st.subheader("Compare the models")
    st.caption("Metrics are from the notebook's 200,000-row holdout test set.")
    comparison = pd.DataFrame(
        {
            "Accuracy": [RESULTS[name]["accuracy"] for name in RESULTS],
            "Fraud precision": [RESULTS[name]["precision"] for name in RESULTS],
            "Fraud recall": [RESULTS[name]["recall"] for name in RESULTS],
            "Fraud F1": [RESULTS[name]["f1"] for name in RESULTS],
        },
        index=list(RESULTS),
    )
    st.dataframe(
        comparison.style.format({column: "{:.1%}" for column in comparison.columns}),
        use_container_width=True,
    )
    st.bar_chart(comparison[["Accuracy", "Fraud precision", "Fraud recall", "Fraud F1"]])

    selected = st.selectbox("Confusion matrix", list(RESULTS), index=1)
    tn, fp = RESULTS[selected]["matrix"][0]
    fn, tp = RESULTS[selected]["matrix"][1]
    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual legitimate", "Actual fraud"],
        columns=["Predicted legitimate", "Predicted fraud"],
    )
    st.dataframe(matrix, use_container_width=True)
    st.caption("False positives: legitimate transactions flagged. False negatives: fraud transactions missed.")
    st.warning(
        "Random Forest catches nearly all fraud in this test set (99% recall), but its 60% fraud "
        "precision means many alerts are false positives. A real system would tune the alert threshold."
    )

else:
    st.subheader("Try a sample transaction")
    st.write("Enter transaction details to see the saved Random Forest model's output.")
    st.warning(
        "Demo limitation: the notebook's IQR outlier bounds were not saved with the model, "
        "so this screen cannot reproduce that training preprocessing step exactly."
    )

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        with col1:
            distance_home = st.number_input("Distance from home", min_value=0.0, value=10.0, step=0.5)
            distance_last = st.number_input(
                "Distance from previous transaction", min_value=0.0, value=5.0, step=0.5
            )
            price_ratio = st.number_input(
                "Purchase price / median purchase price",
                min_value=0.0001,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )
            repeat_retailer = st.selectbox("Repeat retailer?", ["Yes", "No"])
        with col2:
            used_chip = st.selectbox("Chip used?", ["Yes", "No"])
            used_pin = st.selectbox("PIN used?", ["Yes", "No"])
            online_order = st.selectbox("Online order?", ["Yes", "No"])
        submitted = st.form_submit_button("Analyze transaction")

    if submitted:
        binary = [
            float(repeat_retailer == "Yes"),
            float(used_chip == "Yes"),
            float(used_pin == "Yes"),
            float(online_order == "Yes"),
        ]
        price_category = next(
            category
            for upper, category in [(0.5, 0), (1.0, 1), (2.0, 2), (5.0, 3), (10.0, 4)]
            if price_ratio <= upper
        )
        values = {
            "distance_from_home": distance_home,
            "distance_from_last_transaction": distance_last,
            "ratio_to_median_purchase_price": price_ratio,
            "repeat_retailer": binary[0],
            "used_chip": binary[1],
            "used_pin_number": binary[2],
            "online_order": binary[3],
            "home_to_last_ratio": distance_home / (distance_last + 1e-5),
            "transaction_score": sum(binary),
            "price_category": price_category,
        }
        model_input = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
        st.markdown("#### Derived model inputs")
        st.dataframe(model_input, use_container_width=True, hide_index=True)

        try:
            if not MODEL_PATH.exists() or not SCALER_PATH.exists():
                raise FileNotFoundError("model.pkl and scaler.pkl must be beside app.py")
            model, scaler = load_model_artifacts()
            expected = getattr(scaler, "n_features_in_", len(FEATURES))
            if expected != len(FEATURES):
                raise ValueError(f"Saved scaler expects {expected} features; app creates {len(FEATURES)}.")
            scaled_input = scaler.transform(model_input)
            prediction = int(model.predict(scaled_input)[0])
            fraud_probability = float(model.predict_proba(scaled_input)[0][1])
            left, right = st.columns(2)
            if prediction == 1:
                left.error("Model result: flagged as potentially fraudulent")
            else:
                left.success("Model result: classified as likely legitimate")
            right.metric("Estimated fraud probability", f"{fraud_probability:.1%}")
            st.caption("This is a model score for demonstration, not a confirmed fraud determination.")
        except Exception as exc:
            st.error(f"Could not run the model: {exc}")
