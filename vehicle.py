import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    # Convert data types
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Drop rows with missing crucial values
    df.dropna(subset=['make', 'fuel', 'year', 'mileage', 'price'], inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df

df = load_data()

# Encode 'make' and 'fuel' using mapping
make_list = sorted(df['make'].dropna().unique())
fuel_list = sorted(df['fuel'].dropna().unique())

make_map = {name: idx for idx, name in enumerate(make_list)}
fuel_map = {name: idx for idx, name in enumerate(fuel_list)}

df['make_encoded'] = df['make'].map(make_map)
df['fuel_encoded'] = df['fuel'].map(fuel_map)

# Features and target
X = df[['make_encoded', 'year', 'mileage', 'fuel_encoded']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("🚗 Vehicle Price Predictor")
st.write("Enter vehicle specifications below to get a predicted price.")

# Input fields
make_input = st.selectbox("Select Car Brand/Make", make_list)
fuel_input = st.selectbox("Select Fuel Type", fuel_list)

year_input = st.number_input("Year of Manufacture", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=2015)
mileage_input = st.number_input("Mileage (in km)", min_value=0, value=50000)

if st.button("Predict Price"):
    try:
        input_data = pd.DataFrame({
            'make_encoded': [make_map[make_input]],
            'year': [int(year_input)],
            'mileage': [float(mileage_input)],
            'fuel_encoded': [fuel_map[fuel_input]]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Vehicle Price: ₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed due to: {e}")

# Model metrics
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    st.write("📉 Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
    st.write("📈 R² Score:", round(r2_score(y_test, y_pred), 4))
