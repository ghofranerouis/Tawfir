# Tawfir Platform (منصة توفير) 🚀
**Smart AI-Driven Decision Support System for Sustainable Food Management in University Canteens.**

## 🌟 Overview
**Tawfir** (Arabic for *Saving*) is an innovative startup project developed under the Algerian ministerial decree **1275** (Graduation Diploma - Startup). The platform leverages **Artificial Intelligence** and **Machine Learning** to modernize the management of university restaurants. 

By analyzing historical student attendance and external factors like weather, Tawfir provides administrators with precise, data-driven recommendations to reduce food waste, specifically targeting bread and daily meals.

## ✨ Key Features
- **Intelligent Forecasting:** Uses a **Random Forest Regressor** to predict student attendance with high accuracy.
- **Dynamic Variable Integration:** Incorporates academic calendars and real-time weather conditions into the prediction logic.
- **Administrative Decision Box:** Automatically generates a formal recommendation report for procurement officers.
- **Impact Analytics:** Visualizes "Prevented Waste" and financial savings through interactive dashboards.
- **RTL Support:** Fully localized Arabic interface designed for the Algerian administrative context.

## 🧠 Technical Logic (Based on `app.py`)
The system follows a sophisticated data pipeline:
1. **Data Ingestion:** Simulates/Fetches digitized attendance records.
2. **Feature Engineering:** Maps categorical data (Days, Weather) into numerical vectors for the ML model.
3. **The Model:** A `Random Forest` ensemble trained on university flow patterns (e.g., recognizing surges on Mondays/Wednesdays and drops on Thursdays).
4. **Safety Buffer:** Applies a 1.25x coefficient to the AI prediction to ensure meal sufficiency while minimizing surplus.

## 🛠 Tech Stack
- **Language:** Python 3.x
- **Frontend:** Streamlit (Custom CSS for Arabic RTL support)
- **Machine Learning:** Scikit-learn (RandomForestRegressor)
- **Data Science:** Pandas, NumPy
- **Visualizations:** Streamlit Native Charts

## 🔗 Live Prototype:  
[RG Library Prototype](https://rgtawfir.streamlit.app/)

## 📂 Project Structure
```text
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── tawfir/
│   └── logo.png        # Official branding
└── data/               # (Planned) Localized attendance datasets
