import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# AQI Category Function
# ----------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good 😊", "green"
    elif aqi <= 100:
        return "Satisfactory 🙂", "yellow"
    elif aqi <= 200:
        return "Moderate 😐", "orange"
    elif aqi <= 300:
        return "Poor 😷", "red"
    elif aqi <= 400:
        return "Very Poor 🤢", "purple"
    else:
        return "Severe ☠️", "brown"

# ----------------------------
# Load Model and Scaler
# ----------------------------
model = pickle.load(open("aqi_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------
# App Title
# ----------------------------
st.set_page_config(page_title="Mumbai AQI Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center; color:darkblue;'>🌿 Mumbai AQI Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("Enter pollutant levels below and click **Predict AQI**")

# ----------------------------
# Input Fields
# ----------------------------
pm25 = st.number_input("PM2.5", min_value=0.0, max_value=500.0, value=50.0)
pm10 = st.number_input("PM10", min_value=0.0, max_value=500.0, value=100.0)
no2 = st.number_input("NO2", min_value=0.0, max_value=200.0, value=50.0)
so2 = st.number_input("SO2", min_value=0.0, max_value=200.0, value=20.0)
co = st.number_input("CO", min_value=0.0, max_value=20.0, value=1.0)
o3 = st.number_input("O3", min_value=0.0, max_value=200.0, value=50.0)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict AQI"):

    # Prepare input
    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    aqi = prediction[0]

    # Get AQI category & color (color only used for gauge)
    category, color = aqi_category(aqi)

    # ----------------------------
    # Layout Columns
    # ----------------------------
    col1, col2 = st.columns([1, 2])

    # Column 1: AQI Card + Gauge
    with col1:
        # Simple AQI display without colored background
        st.markdown(f"### Predicted AQI: {aqi:.2f}")
        st.markdown(f"### Category: {category}")

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi,
            title={'text': "AQI Gauge"},
            gauge={
                'axis': {'range': [0,500]},
                'bar': {'color': color},
                'steps': [
                    {'range':[0,50], 'color':'green'},
                    {'range':[51,100], 'color':'yellow'},
                    {'range':[101,200], 'color':'orange'},
                    {'range':[201,300], 'color':'red'},
                    {'range':[301,400], 'color':'purple'},
                    {'range':[401,500], 'color':'brown'}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Column 2: Pollutant Radar Chart
    with col2:
        pollutants = ['PM2.5','PM10','NO2','SO2','CO','O3']
        values = [pm25, pm10, no2, so2, co, o3]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=pollutants,
            fill='toself',
            name='Pollutant Levels'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(values)*1.2])),
            showlegend=False,
            title="Pollutant Levels Radar Chart"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ----------------------------
    # Downloadable Prediction Report
    # ----------------------------
    df_report = pd.DataFrame({
        'Pollutant': pollutants + ['AQI', 'Category'],
        'Value': values + [aqi, category]
    })

    csv = df_report.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download Prediction Report as CSV",
        data=csv,
        file_name='aqi_prediction_report.csv',
        mime='text/csv'
    )