import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the trained Random Forest model
model = joblib.load("rf_model.joblib")

# Define NHMRC risk categories
def classify_risk(cfu):
    if cfu < 40:
        return "Safe"
    elif cfu <= 200:
        return "Caution"
    else:
        return "High Risk"

# Title
st.title("Enterococci Prediction Tool")

# Sidebar for single prediction
st.sidebar.header("Single Prediction")
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
month = st.sidebar.slider("Month", 1, 12, datetime.now().month)
season = st.sidebar.selectbox("Season", ["Summer", "Autumn", "Winter", "Spring"])
season_map = {"Summer": 1, "Autumn": 2, "Winter": 3, "Spring": 4}
season_num = season_map[season]

# Predict button
if st.sidebar.button("Predict"):
    input_data = pd.DataFrame([[rainfall, month, season_num]], columns=["Rainfall", "Month", "Season"])
    log_pred = model.predict(input_data)[0]
    pred_cfu = int(np.exp(log_pred))
    risk = classify_risk(pred_cfu)
    st.sidebar.markdown(f"### Predicted Enterococci: {pred_cfu} cfu/100ml")
    st.sidebar.markdown(f"### Risk Category: **{risk}**")

# File upload for batch prediction
st.header("Batch Prediction from File")
uploaded_file = st.file_uploader("Upload CSV or Excel file with 'Date' and 'Rainfall' columns", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        # Extract month and season from date
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.month
        df["Season"] = df["Date"].dt.month % 12 // 3 + 1
        df["Rainfall"] = pd.to_numeric(df["Rainfall"], errors="coerce")
        df.dropna(subset=["Rainfall"], inplace=True)

        # Predict
        input_data = df[["Rainfall", "Month", "Season"]]
        log_preds = model.predict(input_data)
        df["Predicted_cfu"] = np.exp(log_preds).astype(int)
        df["Risk"] = df["Predicted_cfu"].apply(classify_risk)

        # Show results
        st.subheader("Prediction Results")
        st.dataframe(df[["Date", "Rainfall", "Predicted_cfu", "Risk"]])

        # Visualizations
        st.subheader("Rainfall vs Predicted Enterococci")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Rainfall", y="Predicted_cfu", hue="Risk", ax=ax)
        ax.set_ylabel("Predicted Enterococci (cfu/100ml)")
        st.pyplot(fig)

        st.subheader("Time Series of Predicted Enterococci")
        fig2, ax2 = plt.subplots()
        df_sorted = df.sort_values("Date")
        ax2.plot(df_sorted["Date"], df_sorted["Predicted_cfu"], marker="o")
        ax2.set_ylabel("Predicted Enterococci (cfu/100ml)")
        ax2.set_xlabel("Date")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error processing file: {e}")
