import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Page configuration
st.set_page_config(
    page_title="House Cost Prediction",
    page_icon="🏠",
    layout="wide"
)

# Heading
st.title("🏠 House Cost Prediction")
st.write(
    "A Machine Learning project using Linear Regression "
    "to estimate house prices."
)

# Load dataset and train model
@st.cache_data
def load_data():
    return pd.read_csv("houses.csv")

try:
    data = load_data()

    required_columns = [
        "area_sqft", "bedrooms", "bathrooms",
        "age_years", "price"
    ]

    if not all(col in data.columns for col in required_columns):
        st.error("Dataset columns do not match the required format.")
        st.stop()

    data = data[required_columns].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(data) < 10:
        st.error("Please use a dataset with at least 10 valid rows.")
        st.stop()

    X = data[
        ["area_sqft", "bedrooms", "bathrooms", "age_years"]
    ]
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    st.success("Machine Learning model trained successfully!")

    # Model evaluation
    st.header("📊 Model Performance")

    col1, col2 = st.columns(2)

    col1.metric("Mean Absolute Error", f"{mae:,.2f}")
    col2.metric("R² Score", f"{r2:.4f}")

    st.caption(
        "Metrics are calculated on the test data. "
        "R² is not a guaranteed measure of real-world accuracy."
    )

    # House details
    st.header("🏡 Enter House Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq. ft.)",
            min_value=100,
            max_value=100000,
            value=1500,
            step=100
        )

        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=1,
            max_value=20,
            value=3
        )

    with col2:
        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=1,
            max_value=20,
            value=2
        )

        age = st.number_input(
            "House Age (years)",
            min_value=0,
            max_value=200,
            value=10
        )

    if st.button("🔮 Predict House Price", type="primary"):
        new_house = pd.DataFrame([{
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age
        }])

        predicted_price = model.predict(new_house)[0]

        st.subheader("💰 Predicted House Price")
        st.success(f"{predicted_price:,.2f} price units")

        st.caption(
            "This is an estimate from the trained model, "
            "not a guaranteed market price. Confirm the dataset currency."
        )

    # Actual vs predicted graph
    st.header("📈 Actual vs Predicted House Prices")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(y_test, predictions, alpha=0.7)

    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))

    ax.plot([low, high], [low, high], linestyle="--")

    ax.set_xlabel("Actual House Prices")
    ax.set_ylabel("Predicted House Prices")
    ax.set_title("Actual vs Predicted Prices")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Dataset preview
    with st.expander("View Dataset"):
        st.dataframe(data.head(20), use_container_width=True)

except FileNotFoundError:
    st.error(
        "houses.csv was not found. Keep it in the same folder as app.py."
    )

except Exception as e:
    st.error(f"An error occurred: {e}")
    
