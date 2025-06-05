# Required: pip install streamlit pandas scikit-learn numpy

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (ensure this file is present in same folder or use Streamlit uploader)
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.dropna(subset=['price'], inplace=True)
    df.fillna(0, inplace=True)
    return df

df = load_data()

# Label encode manually for make and fuel
make_list = sorted(df['make'].dropna().unique())
fuel_list = sorted(df['fuel'].dropna().unique())

make_map = {name: idx for idx, name in enumerate(make_list)}
fuel_map = {name: idx for idx, name in enumerate(fuel_list)}
reverse_make_map = {v: k for k, v in make_map.items()}
reverse_fuel_map = {v: k for k, v in fuel_map.items()}

# Encode for modeling
df['make_encoded'] = df['make'].map(make_map)
df['fuel_encoded'] = df['fuel'].map(fuel_map)

X = df[['make_encoded', 'year', 'mileage', 'fuel_encoded']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 🎯 Streamlit UI
st.title("🚗 Vehicle Price Predictor")
st.write("Enter vehicle details to predict its estimated market price.")

make_input = st.selectbox("Select Car Brand/Make", make_list)
fuel_input = st.selectbox("Select Fuel Type", fuel_list)
year_input = st.slider("Manufacturing Year", int(df['year'].min()), int(df['year'].max()), 2015)
mileage_input = st.number_input("Mileage (in km)", min_value=0, value=50000)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'make_encoded': [make_map[make_input]],
        'year': [year_input],
        'mileage': [mileage_input],
        'fuel_encoded': [fuel_map[fuel_input]]
    })

    predicted_price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Vehicle Price: ₹ {predicted_price:,.2f}")

# 📊 Show model performance
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    st.write(f"📉 Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"📈 R² Score: {r2_score(y_test, y_pred):.4f}")
