import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")

    # Drop rows with missing required values  
    df.dropna(subset=['make', 'fuel', 'year', 'mileage', 'price'], inplace=True)

    # Clean and convert year, mileage, price columns  
    df['year'] = pd.to_numeric(df['year'], errors='coerce')  
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')  
    df['price'] = pd.to_numeric(df['price'], errors='coerce')  

    # Drop rows with invalid numeric data  
    df.dropna(subset=['year', 'mileage', 'price'], inplace=True)  

    # Convert to integer  
    df['year'] = df['year'].astype(int)  
    df['mileage'] = df['mileage'].astype(int)  
    df['price'] = df['price'].astype(int)

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

# Get min and max year from dataset
min_year = int(df['year'].min())
max_year = int(df['year'].max())
default_year = 2015

# Adjust default_year if out of dataset range
if default_year < min_year:
    default_year = min_year
elif default_year > max_year:
    default_year = max_year

year_input = st.number_input(
    "Year of Manufacture",
    min_value=min_year,
    max_value=max_year,
    value=default_year
)

mileage_input = st.number_input("Mileage (in km)", min_value=0, value=50000)

make_input = st.selectbox("Select Car Brand/Make", make_list)
fuel_input = st.selectbox("Select Fuel Type", fuel_list)

if st.button("Predict Price"):
    try:
        input_data = pd.DataFrame({
            'make_encoded': [make_map[make_input]],
            'year': [int(year_input)],
            'mileage': [int(mileage_input)],
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
