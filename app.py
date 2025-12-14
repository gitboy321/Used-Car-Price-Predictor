import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/car_price_model.pkl")

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Predictor")
st.caption("This tool gives an approximate market value based on historical data.")
st.write("Enter car details to predict the selling price")

# -------------------------
# MAPPINGS (VERY IMPORTANT)
# -------------------------

brand_map = {
    "Maruti": 0,
    "Hyundai": 1,
    "Honda": 2,
    "Toyota": 3,
    "Tata": 4,
    "Mahindra": 5
}

seller_type_map = {
    "Individual": 0,
    "Dealer": 1
}

fuel_type_map = {
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
}

transmission_map = {
    "Manual": 0,
    "Automatic": 1
}

# -------------------------
# USER INPUTS (CLEAN UI)
# -------------------------

brand = st.selectbox("Brand", list(brand_map.keys()))
vehicle_age = st.slider("Vehicle Age (Years)", 0, 25)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)

seller_type = st.selectbox("Seller Type", list(seller_type_map.keys()))
fuel_type = st.selectbox("Fuel Type", list(fuel_type_map.keys()))
transmission = st.selectbox("Transmission Type", list(transmission_map.keys()))

mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=40.0)
engine = st.number_input("Engine (CC)", min_value=600, max_value=5000)
max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=500.0)
seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8])

# -------------------------
# PREDICTION
# -------------------------

if st.button("Predict Price"):
    input_data = np.array([[
        brand_map[brand],
        vehicle_age,
        km_driven,
        seller_type_map[seller_type],
        fuel_type_map[fuel_type],
        transmission_map[transmission],
        mileage,
        engine,
        max_power,
        seats
    ]])

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Selling Price: ₹ {int(prediction[0]):,}")
