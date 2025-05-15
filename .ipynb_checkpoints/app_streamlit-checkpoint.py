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
from nbformat.v4 import new_code_cell
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import streamlit.components.v1 as components
from feature_engineer import add_features

# ── 1. GLOBAL CONFIG & CSS ────────────────────────────────
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
      color: black !important; /* força contraste */
    }
    .element-container:has(.stMetric) {
      display: flex;
      justify-content: center;
    }
  </style>
""", unsafe_allow_html=True)


# ── 2. PAGE NAVIGATION ─────────────────────────────────────
pages = ["Home", "Dataset", "Notebook"]
choice = st.sidebar.radio("Go to", pages)

# ── 3. MODEL & DATA LOADING ────────────────────────────────
@st.cache_resource
def load_model():
    IS_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"
    local_path = "car_price_stacked_pipeline.pkl"

    if IS_CLOUD:
        url = "https://drive.google.com/uc?export=download&id=1cpMyNSxLTBVixk-2BrjBlZP8nDd6-6YH"
        if not os.path.exists(local_path):
            with st.spinner("🔽 Baixando modelo do Google Drive..."):
                response = requests.get(url)
                with open(local_path, "wb") as f:
                    f.write(response.content)
                st.write("✅ Modelo baixado com sucesso.")
                st.write(f"Tamanho do arquivo: {os.path.getsize(local_path)} bytes")

    # Verifica se o arquivo existe e tem tamanho mínimo esperado (~1MB)
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1000000:
        raise RuntimeError("🚨 Modelo não foi baixado corretamente. Arquivo corrompido ou vazio.")

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

# ── 4. HOME PAGE ────────────────────────────────────────────
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
                st.metric("Estimated Price", f"$ {price:,.2f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ── 5. DATASET PAGE ─────────────────────────────────────────
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

# ── 6. NOTEBOOK PAGE (with executed outputs) ──────────────────
else:
    st.title("Executed Notebook")
    st.write("Rendering `vehicle-prediction-tool.ipynb` with its outputs below:")

    try:
        nb = nbformat.read("prediction-tool-final.ipynb", as_version=4)
        nb.cells.insert(0, new_code_cell("%matplotlib inline"))
        client = NotebookClient(nb, timeout=600, allow_errors=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
            client.execute()
        html_exporter = HTMLExporter()
        body, _ = html_exporter.from_notebook_node(nb)
        components.html(body, height=800, scrolling=True)

    except FileNotFoundError:
        st.error("`vehicle-prediction-tool.ipynb` not found.")
    except Exception as e:
        st.error(f"Error running notebook: {e}")
