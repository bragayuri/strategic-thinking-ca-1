# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import warnings
import os
import requests

import nbformat
from nbconvert import HTMLExporter
import streamlit.components.v1 as components
from feature_engineer import add_features
import gdown

# 1.GENERAL CONFIG AND LAYOUT DESIGN
st.set_page_config(page_title="Used Car Price Tool", layout="wide", page_icon="🚘")
st.markdown("""
  <style>
    .css-18e3th9 { padding-top: 1rem; }
    .css-1d391kg { font-size:2rem; font-weight:bold;}
    .stMetric {
      background-color: #f8f9fa;
      border-radius: 8px;
      padding: 1rem;
      width: fit-content !important;
      margin: 1rem auto !important;
      color: black !important; 
    }
    .element-container:has(.stMetric) {
      display: flex;
      justify-content: center;
    }
  </style>
""", unsafe_allow_html=True)

# 2. PAGE NAV
pages = ["Home", "Dataset", "Notebook"]
choice = st.sidebar.radio("Go to", pages)

# 3. LOAD DATA AND MODEL
@st.cache_resource
def load_model():
    local_path = "car_price_stacked_pipeline.pkl"
    file_id = "1cpMyNSxLTBVixk-2BrjBlZP8nDd6-6YH"
    url = f"https://drive.google.com/uc?id={file_id}"

    # Se já baixou uma vez, não baixa de novo
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1000000:
        with st.spinner("🔽 Baixando modelo do Google Drive via gdown..."):
            gdown.download(url, local_path, quiet=False)
            st.success("✅ Modelo baixado com sucesso.")
            st.info(f"Tamanho: {os.path.getsize(local_path)} bytes")

    return joblib.load(local_path)



@st.cache_data
def load_options():
    df = pd.read_csv("new_vehicle_all_price.csv")
    df.rename(columns={
        "manufacturer": "make",
        "fuel": "fuel_type",
        "odometer": "Kilometer",  
        "price": "price"
    }, inplace=True)
    df = df[['make','model','fuel_type','transmission','year']].dropna()
    for c in ['make','model','fuel_type','transmission']:
        df[c] = df[c].str.title()
    return df

df_valid = load_options()
model    = load_model()

# 4. MAIN PAGE 
if choice == "Home":
    with st.sidebar:
        st.header("Vehicle Details")
        make  = st.selectbox("Make", sorted(df_valid.make.unique()))
        mod   = st.selectbox("Model", sorted(df_valid[df_valid.make==make].model.unique()))
        fuel  = st.selectbox("Fuel Type", sorted(df_valid[(df_valid.make==make)&(df_valid.model==mod)].fuel_type.unique()))
        trans = st.selectbox("Transmission", sorted(df_valid[(df_valid.make==make)&(df_valid.model==mod)].transmission.unique()))
        year  = st.number_input("Year", 1970, datetime.datetime.now().year, 2020)
        km    = st.number_input("Mileage (km)", 0, 500_000, 40_000)
        go    = st.button("Calculate Price")

    st.title("Used Car Price Estimator")
    st.write("Fill the sidebar and hit **Calculate Price**.")

    if go:
        errs = []
        if not (1970 <= year <= datetime.datetime.now().year):
            errs.append("Year out of range.")
        if km < 0:
            errs.append("Mileage must be ≥ 0.")
        if errs:
            for e in errs:
                st.warning(e)
        else:
            df_in = pd.DataFrame([{
                "make": make.lower(),
                "model": mod.lower(),
                "fuel_type": fuel.lower(),
                "transmission": trans.lower(),
                "year": year,
                "Kilometer": km
            }])
            try:
                logp  = model.predict(df_in)[0]
                price = np.expm1(logp)
                price_display = f"${price:,.2f}"
                st.markdown(f"""
                    <div style="
                       display: flex;
                       justify-content: start;
                       margin-top: 2rem;
                    ">
                        <div style="
                            background-color: #111;
                            padding: 1.5rem 3rem;
                            border-radius: 12px;
                            color: white;
                            font-size: 2.5rem;
                            font-weight: bold;
                            box-shadow: 0 0 12px rgba(0,255,174,0.1);
                        ">
                            Estimated Price: {price_display}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# 5. DATA PAGE
elif choice == "Dataset":
    st.title("Dataset Description")
    st.write("""
    **Fields**:  
    - make, model, fuel_type, transmission, year, Kilometer, price
    """)
    df = pd.read_csv("new_vehicle_all_price.csv")
    st.subheader("Sample Records")
    st.dataframe(df.sample(5), use_container_width=True)
    st.subheader("Summary Statistics")
    st.write(df.describe())

# 6.NOTEBOOK PAGE
else:
    st.title("Executed Notebook")
    st.write("Rendering `vehicle-prediction-tool.ipynb` with its outputs below:")

    try:
        with open("prediction-tool-final.ipynb") as f:
            nb = nbformat.read(f, as_version=4)
        html_exporter = HTMLExporter()
        (body, _) = html_exporter.from_notebook_node(nb)
        components.html(body, height=1000, scrolling=True)

    except FileNotFoundError:
        st.error("`vehicle-prediction-tool.ipynb` not found.")
    except Exception as e:
        st.error(f"Error loading notebook: {e}")
