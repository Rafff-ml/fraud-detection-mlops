import streamlit as st
import requests
import numpy as np 


st.title("Fintech Fraud Detection Dashboard")


amount = st.number_input("Transaction Amount", value=100.0)
hour = st.slider("hour", 0, 23, 12)
tx_count = st.slider("Transaction last 10", 0, 20, 2)
avg_amount = st.number_input("Average recent amount", value=80.0)


if st.button("check Fraud"):

    amount_log = np.log1p(amount)
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    deviation = amount - avg_amount 


    payload = {
        "Amount": amount,
        "hour": hour,
        "amount_log": amount_log,
        "is_night": is_night,
        "tx_count_last_10": tx_count,
        "avg_amount_last_10": avg_amount,
        "amount_deviation": deviation
    }


    response = requests.post(
        "https://fraud-detection-mlops-i1ib.onrender.com/",
        json=payload
    )

    result = response.json()


    st.write(result)