import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Vehicle Price Prediction Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("dataset.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Data Cleaning
    st.subheader("Cleaning Data")
    df.dropna(inplace=True)  # Drop rows with missing values
    st.write(f"Rows after removing NA: {df.shape[0]}")

    # Encode categorical variables
    st.subheader("Encoding categorical columns")
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
        st.write(f"Encoded: {col}")

    # Feature-Target split
    if "price" not in df.columns:
        st.error("Dataset must contain a 'price' column for prediction.")
    else:
        X = df.drop("price", axis=1)
        y = df["price"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Show performance
        st.subheader("Model Performance")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

        # Prediction Interface
        st.subheader("Predict Vehicle Price")

        user_input = {}
        for feature in X.columns:
            if df[feature].nunique() < 20:
                user_input[feature] = st.selectbox(f"{feature}:", sorted(df[feature].unique()))
            else:
                user_input[feature] = st.slider(f"{feature}:", float(df[feature].min()), float(df[feature].max()))

        if st.button("Predict Price"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Price: ₹{prediction:,.2f}")

        # Visualizations
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader("Price Distribution")
        fig2 = px.histogram(df, x='price', nbins=40, title="Price Distribution")
        st.plotly_chart(fig2)

        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        fig3 = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance")
        st.plotly_chart(fig3)
else:
    st.info("Please upload a dataset to continue.")
