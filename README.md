
# 🚗 Used Car Price Prediction Tool

A machine learning solution to **predict used car trade-in prices** for dealerships. This tool uses ensemble learning with **Random Forest** and **Histogram-based Gradient Boosting**, wrapped in a clean interface powered by **Streamlit** and **FastAPI**.

It allows dealerships to automate pricing decisions, reduce appraisal times from minutes to seconds, and ensure fair and competitive quotes based on real market data.

---

## 🧠 Project Overview

- 📊 Predicts vehicle prices based on features like brand, age, mileage, fuel type, etc.
- ⚙️ Built using Python, Scikit-Learn, FastAPI, and Streamlit.
- 🧪 Trained on 20,000 vehicle records (sampled from 450,000+).
- 🧮 Final model: Stacked Ensemble → RMSE: **0.38**, R²: **0.765**
- 🚀 Delivered as a white-label-ready tool for dealership integration.

---

## 📁 Project Structure


.
├── prediction-tool-final.ipynb     # Jupyter Notebook with data pipeline and model training
├── app/                           
│   ├── main.py                     # FastAPI backend for predictions
│   ├── model.joblib                # Serialized model
├── ui/
│   ├── streamlit_app.py            # Streamlit frontend (optional)
├── requirements.txt               # Python dependencies

---

## 🔧 Setup Instructions

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the FastAPI backend**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

3. **(Optional) Run the Streamlit interface**
   ```bash
    streamlit run streamlit_app.py
   ```

4. **Make predictions via API**
   - Endpoint: `POST /predict`
   - Example request:
     ```json
     {
       "brand": "Toyota",
       "model": "Corolla",
       "year": 2018,
       "mileage": 85000,
       "fuel": "Petrol",
       "transmission": "Automatic"
     }
     ```

---

## 🛠️ Engineered Features

- `age`: Vehicle age (based on current year)
- `mileage_per_year`: Adjusted mileage normalized by age
- `is_luxury_brand`: Flag for premium brands (e.g., BMW, Audi, Mercedes)

These features significantly improved model performance and business value.

---

## 📈 Model Performance

- **Model Type**: Stacked Ensemble
  - Base learners: Random Forest, Histogram-based Gradient Boosting
  - Meta-learner: RidgeCV
- **Test RMSE**: 0.38
- **Test R²**: 0.765
- Predicts prices within ±10% of true values

---

## 🌱 Future Enhancements

- Rebuild UI as embeddable Web Component
- Expand dataset to full 450K+ listings
- Include service history or location-based pricing
- Containerized deployment on cloud services

---

## 📜 License

This project is for educational and prototyping purposes only. Dataset used under Gigasheet's sample data terms of use. For commercial use, please verify licensing.

---

Made with by Yuri Braga
