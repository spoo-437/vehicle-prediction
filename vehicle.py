# Install these if not already:
# pip install streamlit pandas scikit-learn matplotlib seaborn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load data — relative path for deployment safety
df = pd.read_csv("dataset.csv")


# Encode categorical variables
label_enc_make = LabelEncoder()
df['make'] = label_enc_make.fit_transform(df['make'])

label_enc_fuel = LabelEncoder()
df['fuel'] = label_enc_fuel.fit_transform(df['fuel'])

# Drop rows with missing target
df.dropna(subset=['price'], inplace=True)

# Fill remaining missing values
df.fillna(0, inplace=True)

# Features & target
X = df[['make', 'year', 'mileage', 'fuel']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("🚗 Vehicle Price Predictor")

# User inputs
make_input = st.selectbox("Select Make:", df['make'].unique())
year_input = st.slider("Select Year of Manufacture:", int(df['year'].min()), int(df['year'].max()), 2015)
mileage_input = st.number_input("Enter Mileage (in km):", min_value=0, value=50000)
fuel_input = st.selectbox("Select Fuel Type:", df['fuel'].unique())

# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'make': [make_input],
        'year': [year_input],
        'mileage': [mileage_input],
        'fuel': [fuel_input]
    })

    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")

# Model performance metrics
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R² Score:", r2_score(y_test, y_pred))
