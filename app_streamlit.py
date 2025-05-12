# app_streamlit.py

# ── Importing Libraries ─────────────────────────────────────
import streamlit as st  # For creating the web app UI
import pandas as pd     # For data manipulation
import numpy as np      # For numerical operations
import joblib           # For loading the trained model
import datetime         # For working with dates
import warnings         # For handling warnings

# For loading and executing Jupyter Notebooks
import nbformat
from nbformat.v4 import new_code_cell
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import streamlit.components.v1 as components

# ── 1. GLOBAL CONFIG & CSS ────────────────────────────────
# Sets up the Streamlit page configuration and custom CSS for styling
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
    }
  </style>
""", unsafe_allow_html=True)

# ── 2. PAGE NAVIGATION ─────────────────────────────────────
# Sidebar radio button to switch between different app pages
pages = ["Home", "Dataset", "Notebook"]
choice = st.sidebar.radio("Go to", pages)

# ── 3. MODEL & DATA LOADING ────────────────────────────────
# Load the machine learning model (cached for performance)
@st.cache_data
def load_model():
    return joblib.load("car_price_full_pipeline.pkl")

# Load and clean the dataset for dropdown options (cached)
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

# Load data and model at runtime
df_valid = load_options()
model    = load_model()

# ── 4. HOME PAGE ────────────────────────────────────────────
# UI for the Home page where user inputs car details and gets a price prediction
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

    # If user clicks the button, validate inputs and make prediction
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
            # Create input DataFrame from user selections
            df_in = pd.DataFrame([{
                "make": make.lower(),
                "model": mod.lower(),
                "fuel_type": fuel.lower(),
                "transmission": trans.lower(),
                "year": year,
                "Kilometer": km
            }])
            try:
                # Make prediction and show result
                logp  = model.predict(df_in)[0]
                price = np.expm1(logp)
                st.metric("Estimated Price", f"$ {price:,.2f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ── 5. DATASET PAGE ─────────────────────────────────────────
# UI for viewing the dataset and its basic statistics
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

# ── 6. NOTEBOOK PAGE (with executed outputs) ────────────────
# UI for rendering a Jupyter Notebook with outputs inside the Streamlit app
else:
    st.title("Executed Notebook")
    st.write("Rendering `vehicle-prediction-tool.ipynb` with its outputs below:")

    try:
        # Load and execute the notebook, then render it as HTML
        nb = nbformat.read("vehicle-prediction-tool.ipynb", as_version=4)
        nb.cells.insert(0, new_code_cell("%matplotlib inline"))
        client = NotebookClient(nb, timeout=600, allow_errors=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
            client.execute()
        html_exporter = HTMLExporter()
        body, _ = html_exporter.from_notebook_node(nb)
        components.html(body, height=800, scrolling=True)

    # Handle errors like missing notebook or execution issues
    except FileNotFoundError:
        st.error("`vehicle-prediction-tool.ipynb` not found.")
    except Exception as e:
        st.error(f"Error running notebook: {e}")
