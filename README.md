# 🛵 SwiftPulse ETA | AI-Powered Logistics & Delivery Risk Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Regressor-orange.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end Machine Learning operations platform for **last-mile food delivery ETA estimation** and **real-time dispatch delay risk classification** (inspired by ops dashboards at Swiggy and Uber Eats).

---

## 📌 Executive Overview

Accurate Estimated Time of Arrival (ETA) is a critical operational KPI for on-demand logistics platforms. Inaccurate estimations result in cold food, driver idle time, SLA penalties, and customer churn. 

**SwiftPulse ETA** processes 20 dynamic variables spanning route geometry, real-time traffic signals, weather conditions, kitchen load, and rider experience to produce sub-10ms delivery predictions. Orders predicted to exceed 90 minutes are classified with **High Delay Risk**, enabling proactive dispatch interventions and route optimization.

### 🌟 Key Highlights
- **High-Precision ML Pipeline**: Scikit-Learn `ColumnTransformer` with median/mode imputation and one-hot encoding paired with an optimized `XGBRegressor`.
- **FastAPI Microservice**: High-throughput REST API featuring Pydantic data validation, async lifespan management, and SQLite persistence.
- **Operations Console**: Modern dark-mode dashboard styled with glassmorphism cards, Google Fonts typography, interactive widgets, and live SQLite telemetry logging.
- **Dual-Mode Hybrid Architecture**: Fully decoupled for production microservices while seamlessly falling back to in-process model execution when deployed on serverless platforms such as **Streamlit Community Cloud**.

---

## 📊 Model Performance & Results

The regression model was trained and evaluated on an 80/20 train-test split across thousands of historical dispatch events:

| Model Architecture | Test MAE (Mean Absolute Error) | Test R² Score | Deployment Status |
| :--- | :---: | :---: | :---: |
| **XGBoost Regressor (Tuned)** | **2.52 mins** | **0.9917** | 🏆 **Production Active** |
| Random Forest Regressor | 2.84 mins | 0.9882 | Baseline Benchmark |

- **Mean Absolute Error (MAE)**: **2.52 minutes** on unseen test dispatches.
- **Coefficient of Determination ($R^2$)**: **0.9917**, capturing over 99.1% of delivery variance.
- **Delay Risk Threshold**: Any prediction exceeding **90.0 minutes** automatically raises a dispatch alert (`🚨 High Delay Risk Detected!`).

---

## 🗂️ Features Engineered

The pipeline ingests 20 structured features across spatial, environmental, kitchen, and human factors:

| Category | Features |
| :--- | :--- |
| **Spatial & Route** | `Road_Distance_km`, `Number_of_Signals`, `Pickup_Zone`, `Dropoff_Zone`, `Delivery_Distance_Category` |
| **Traffic & Environment** | `Traffic_Level` (Low, Moderate, High, Jam, Severe), `Weather` (Clear, Rainy, Stormy, Foggy, Windy), `Vehicle_Type` (Bike, Scooter, Electric Scooter, Bicycle, Car) |
| **Kitchen & Order** | `Preparation_Time_Min`, `Restaurant_Load` (Low, Medium, High), `Cuisine_Type`, `Restaurant_Rating`, `Order_Items` |
| **Rider & Timing** | `Rider_Experience_Years`, `Rider_Rating`, `Order_Hour` (0-23), `Day_of_Week`, `Is_Weekend`, `Is_Festival`, `Delivery_Priority` |

---

## 🏛️ System Architecture

```
[ Client / Browser ]
        │
        ▼
[ Streamlit Ops Dashboard ] ─── (app/dashboard.py)
        │
   ┌────┴───────────────────────────────────────┐
   │                                            │
   ▼ (HTTP / JSON)                              ▼ (Fallback on Cloud)
[ FastAPI Microservice ] (src/api.py)     [ In-Process Model ]
   │                                            │ (models/eta_model.joblib)
   ▼                                            ▼
[ SQLite Persistence Engine ] ─────────── (logs/predictions.db)
```

1. **Microservice Mode (Local / Production Docker)**: Streamlit sends asynchronous POST requests to `http://127.0.0.1:8000/predict` and retrieves dispatch history via `GET /history`.
2. **Seamless Cloud Mode (Streamlit Community Cloud)**: If the external FastAPI port is unreachable, the dashboard automatically loads `models/eta_model.joblib` into memory and reads/writes to `src/database.py` directly without raising connection errors.

---

## 📂 Repository Structure

```text
food-delivery-eta/
├── .gitignore               # Ignores .venv, cache, and large raw data; tracks models/
├── .streamlit/
│   └── config.toml          # Dark theme aesthetics (palette, baseRadius, typography)
├── app/
│   └── dashboard.py         # Streamlit logistics ops console (dual-mode)
├── data/
│   └── delivery_data.csv    # Raw training dataset (gitignored)
├── logs/
│   └── predictions.db       # Local SQLite prediction transaction log
├── models/
│   └── eta_model.joblib     # Serialized best XGBoost pipeline (tracked)
├── src/
│   ├── api.py               # FastAPI microservice REST endpoints
│   ├── database.py          # SQLite schema initialization and CRUD queries
│   ├── test_api.py          # Automated verification test suite
│   └── train.py             # Model training, hyperparameter tuning & evaluation
├── requirements.txt         # Production Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/<your-username>/food-delivery-eta.git
cd food-delivery-eta

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Retrain the Machine Learning Model
To retrain the model from `data/delivery_data.csv`:
```bash
python src/train.py
```
This evaluates both Random Forest and XGBoost, outputs comparison metrics, and serializes the winning pipeline to `models/eta_model.joblib`.

### 5. Run Backend Verification Tests
```bash
python src/test_api.py
```

### 6. Launch Applications

#### Option A: Full Microservice Architecture (Recommended for Local Dev)
In terminal 1 (start FastAPI):
```bash
uvicorn src.api:app --reload --port 8000
```
API Documentation will be accessible at: `http://127.0.0.1:8000/docs`

In terminal 2 (start Streamlit):
```bash
streamlit run app/dashboard.py
```

#### Option B: Standalone Dashboard (Streamlit Cloud Simulation)
Simply launch Streamlit directly without starting FastAPI:
```bash
streamlit run app/dashboard.py
```
The application will automatically detect that the API microservice is offline and run in-process using the pre-trained `models/eta_model.joblib`.

---

## ☁️ Streamlit Community Cloud Deployment

1. Push your repository to GitHub (ensure `models/eta_model.joblib` and `requirements.txt` are committed).
2. Log in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app** and select your repository.
4. Set the **Main file path** to:
   ```text
   app/dashboard.py
   ```
5. Click **Deploy**. The app will run in standalone embedded mode seamlessly.

---

## 📡 REST API Reference

### Health Check
`GET /`
```json
{
  "status": "ok",
  "message": "Food Delivery ETA Prediction API is running"
}
```

### Predict Delivery ETA
`POST /predict`

**Request Body Sample:**
```json
{
  "Order_Hour": 19,
  "Is_Weekend": 1,
  "Is_Festival": 0,
  "Rider_Experience_Years": 4.5,
  "Rider_Rating": 4.8,
  "Restaurant_Rating": 4.5,
  "Order_Items": 3,
  "Preparation_Time_Min": 20,
  "Road_Distance_km": 8.2,
  "Number_of_Signals": 4,
  "Day_of_Week": "Saturday",
  "Weather": "Clear",
  "Pickup_Zone": "CBD",
  "Dropoff_Zone": "Residential",
  "Vehicle_Type": "Bike",
  "Cuisine_Type": "Biryani",
  "Restaurant_Load": "High",
  "Delivery_Distance_Category": "Medium",
  "Traffic_Level": "Moderate",
  "Delivery_Priority": "Express"
}
```

**Response Sample:**
```json
{
  "estimated_minutes": 31.45,
  "delay_risk": "Normal",
  "record_id": 16,
  "message": "Prediction successfully generated and saved to database"
}
```

### Dispatch History
`GET /history?limit=10`
Returns the 10 most recent prediction records persisted in SQLite.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
