import os
import sys
import streamlit as st
import requests
import pandas as pd
import joblib

# Ensure repository root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.database import init_db, insert_prediction, get_recent_predictions

# Initialize SQLite database schema if not present
init_db()

# Page configuration
st.set_page_config(
    page_title="SwiftPulse ETA | AI Logistics",
    page_icon="🛵",
    layout="wide"
)

API_BASE_URL = "http://127.0.0.1:8000"

@st.cache_resource(show_spinner=False)
def load_fallback_model():
    """Load trained XGBoost pipeline directly when FastAPI is unreachable (e.g., on Streamlit Cloud)."""
    model_path = os.path.join(BASE_DIR, "models", "eta_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def check_api_status():
    """Check if the local FastAPI microservice is online."""
    try:
        res = requests.get(f"{API_BASE_URL}/", timeout=0.6)
        return res.status_code == 200
    except Exception:
        return False

# Inject modern CSS with clean typography, glassmorphism, and gradient aesthetics
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

/* Typography base */
html, body, [class*="css"], [class*="st-"], .stMarkdown {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

h1, h2, h3, h4, h5, h6, .hero-title, .panel-title, .section-title {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Background gradient styling */
.stApp {
    background-color: #0B0F19;
    background-image: 
        radial-gradient(circle at 10% 8%, rgba(255, 90, 31, 0.08) 0%, transparent 35%),
        radial-gradient(circle at 90% 85%, rgba(20, 30, 55, 0.3) 0%, transparent 45%);
    background-attachment: fixed;
}

/* Hero Banner Container */
div[class*="st-key-hero_banner"] {
    background: linear-gradient(135deg, rgba(255, 90, 31, 0.16) 0%, rgba(19, 26, 42, 0.85) 50%, rgba(11, 15, 25, 0.95) 100%) !important;
    border: 1px solid rgba(255, 90, 31, 0.25) !important;
    border-radius: 16px !important;
    box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    padding: 1.5rem 2rem !important;
    margin-bottom: 1.25rem !important;
}

/* Glassmorphism Cards */
div[class*="st-key-order_config_card"],
div[class*="st-key-prediction_panel_card"],
div[class*="st-key-history_card"] {
    background: rgba(19, 26, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5), 0 0 1px 1px rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    padding: 1.5rem !important;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
}

div[class*="st-key-order_config_card"]:hover,
div[class*="st-key-prediction_panel_card"]:hover,
div[class*="st-key-history_card"]:hover {
    border-color: rgba(255, 90, 31, 0.22) !important;
    box-shadow: 0 14px 35px -5px rgba(0, 0, 0, 0.6), 0 0 15px rgba(255, 90, 31, 0.06) !important;
}

/* Primary CTA Button - Vibrant Gradient & Hover Lift */
button[kind="primary"],
button[data-testid="stBaseButton-primary"],
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #FF5A1F 0%, #FF8A00 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.75rem !important;
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 18px rgba(255, 90, 31, 0.4) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    width: 100% !important;
}

button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #FF6830 0%, #FF991A 100%) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 28px rgba(255, 90, 31, 0.6), 0 0 15px rgba(255, 138, 0, 0.3) !important;
}

button[kind="primary"]:active,
button[data-testid="stBaseButton-primary"]:active,
div[data-testid="stButton"] button[kind="primary"]:active {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(255, 90, 31, 0.4) !important;
}

/* Secondary Button (Refresh) */
button[kind="secondary"],
button[data-testid="stBaseButton-secondary"],
div[data-testid="stButton"] button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.04) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover,
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 90, 31, 0.4) !important;
    color: #FF8A00 !important;
    transform: translateY(-1px) !important;
}

/* Live Status Pill & Glowing Dot */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 1.1rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34D399;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
}

.status-pulse-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
    animation: status-pulse-anim 2s infinite ease-in-out;
}

@keyframes status-pulse-anim {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.08); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Hero Typography & Elements */
.hero-brand-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.35rem;
}

.hero-icon-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: rgba(255, 90, 31, 0.2);
    border: 1px solid rgba(255, 90, 31, 0.4);
    border-radius: 10px;
    font-size: 1.25rem;
}

.hero-brand-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 1.35rem;
    background: linear-gradient(135deg, #FFFFFF 30%, #FF8A00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}

.hero-version-tag {
    background: rgba(255, 255, 255, 0.08);
    color: #94A3B8;
    border: 1px solid rgba(255, 255, 255, 0.12);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
}

.hero-title {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: #F8FAFC !important;
    margin: 0.2rem 0 !important;
    letter-spacing: -0.02em !important;
}

.hero-subtitle {
    color: #94A3B8 !important;
    font-size: 0.92rem !important;
    margin: 0 !important;
    font-weight: 400 !important;
}

.hero-status-container {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.6rem;
    justify-content: center;
    height: 100%;
}

.hero-meta-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-size: 0.78rem;
    color: #94A3B8;
    font-weight: 500;
}

/* Panel Headers */
.panel-header {
    margin-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 0.75rem;
}

.panel-badge {
    display: inline-block;
    background: rgba(255, 90, 31, 0.15);
    color: #FF8A00;
    border: 1px solid rgba(255, 90, 31, 0.3);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    margin-bottom: 0.4rem;
}

.panel-title {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    margin: 0 0 0.25rem 0 !important;
}

.panel-subtitle {
    font-size: 0.85rem !important;
    color: #64748B !important;
    margin: 0 !important;
}

.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.flags-header {
    font-size: 0.88rem;
    font-weight: 600;
    color: #94A3B8;
    margin-top: 0.75rem;
    margin-bottom: 0.25rem;
}

/* Standby & Telemetry Displays */
.standby-box {
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: rgba(11, 15, 25, 0.4);
    border: 1px dashed rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    margin: 1rem 0;
}

.standby-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    filter: drop-shadow(0 0 12px rgba(255, 90, 31, 0.3));
}

.standby-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.5rem;
}

.standby-desc {
    font-size: 0.85rem;
    color: #64748B;
    line-height: 1.5;
    max-width: 320px;
    margin: 0 auto 1.25rem auto;
}

.standby-pills {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.standby-pill {
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #94A3B8;
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
}

.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-top: 1.25rem;
}

.telemetry-chip {
    background: rgba(11, 15, 25, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.telemetry-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #64748B;
}

.telemetry-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #E2E8F0;
}

/* Metric card styling */
div[data-testid="stMetric"] {
    background: rgba(11, 15, 25, 0.55) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.2rem !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    color: #FF8A00 !important;
}

/* Widget labels & inputs */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #94A3B8 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: rgba(11, 15, 25, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
}
</style>
"""

st.html(CUSTOM_CSS)

# Header Hero Banner Container
is_api_connected = check_api_status()
status_label = "● API Connected" if is_api_connected else "● API Connected (Embedded ML)"
meta_label = "⚡ FastAPI Microservice" if is_api_connected else "⚡ In-Process XGBoost Pipeline"

with st.container(key="hero_banner"):
    banner_col1, banner_col2 = st.columns([3, 1], vertical_alignment="center")
    with banner_col1:
        st.markdown(
            """
            <div class="hero-brand-row">
                <span class="hero-icon-badge">🛵</span>
                <span class="hero-brand-name">SwiftPulse ETA</span>
                <span class="hero-version-tag">AI Logistics Ops</span>
            </div>
            <h1 class="hero-title">Real-Time Dispatch & ETA Intelligence</h1>
            <p class="hero-subtitle">Real-time Machine Learning ETA & Risk Classification powered by XGBoost & FastAPI</p>
            """,
            unsafe_allow_html=True
        )
    with banner_col2:
        st.markdown(
            f"""
            <div class="hero-status-container">
                <div class="status-pill">
                    <span class="status-pulse-dot"></span>
                    <span>{status_label}</span>
                </div>
                <div class="hero-meta-badge">
                    <span>{meta_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Layout: Order Configuration (Left) & Live ETA Prediction (Right)
col_config, col_pred = st.columns([1.18, 0.82], gap="large")

with col_config:
    with st.container(key="order_config_card"):
        st.markdown(
            """
            <div class="panel-header">
                <div class="panel-badge">DISPATCH CONTROLS</div>
                <h3 class="panel-title">📦 Order & Route Logistics</h3>
                <p class="panel-subtitle">Configure fleet routing, kitchen preparation, and environmental conditions</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        cfg_col1, cfg_col2 = st.columns(2)

        with cfg_col1:
            st.markdown('<div class="section-title">🛵 Route & Environment</div>', unsafe_allow_html=True)
            road_distance_km = st.slider("Road_Distance_km", min_value=0.1, max_value=80.0, value=12.5, step=0.1)
            traffic_level = st.selectbox("Traffic_Level", ["Low", "Moderate", "High", "Jam", "Severe"], index=1)
            number_of_signals = st.slider("Number_of_Signals", min_value=0, max_value=30, value=6, step=1)
            vehicle_type = st.selectbox("Vehicle_Type", ["Bike", "Scooter", "Electric Scooter", "Bicycle", "Car"], index=0)
            weather = st.selectbox("Weather", ["Clear", "Rain", "Rainy", "Cloudy", "Storm", "Stormy", "Fog", "Foggy", "Windy"], index=0)
            
            pickup_zone = st.selectbox("Pickup_Zone", ["CBD", "Commercial", "Residential", "Industrial", "Suburban"], index=0)
            dropoff_zone = st.selectbox("Dropoff_Zone", ["Residential", "CBD", "Commercial", "Industrial", "Suburban"], index=0)
            delivery_distance_category = st.selectbox("Delivery_Distance_Category", ["Medium", "Short", "Long"], index=0)
            delivery_priority = st.selectbox("Delivery_Priority", ["Normal", "Express", "Priority", "VIP"], index=0)

        with cfg_col2:
            st.markdown('<div class="section-title">🍳 Kitchen & Rider</div>', unsafe_allow_html=True)
            preparation_time_min = st.slider("Preparation_Time_Min", min_value=1, max_value=60, value=15, step=1)
            restaurant_load = st.selectbox("Restaurant_Load", ["Low", "Medium", "High"], index=1)
            cuisine_type = st.selectbox("Cuisine_Type", ["North Indian", "South Indian", "Biryani", "Pizza", "Burger", "Chinese", "Desserts", "Bakery", "Cafe"], index=0)
            restaurant_rating = st.slider("Restaurant_Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
            order_items = st.number_input("Order_Items", min_value=1, max_value=20, value=2, step=1)
            
            order_hour = st.slider("Order_Hour", min_value=0, max_value=23, value=12, step=1)
            day_of_week = st.selectbox("Day_of_Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=0)
            rider_experience_years = st.slider("Rider_Experience_Years", min_value=0.0, max_value=20.0, value=3.5, step=0.1)
            rider_rating = st.slider("Rider_Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)

            st.markdown('<div class="flags-header">📅 Calendar Flags</div>', unsafe_allow_html=True)
            flag_col1, flag_col2 = st.columns(2)
            with flag_col1:
                is_weekend_cb = st.checkbox("Is_Weekend", value=False)
            with flag_col2:
                is_festival_cb = st.checkbox("Is_Festival", value=False)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Predict Delivery ETA", type="primary", use_container_width=True)

# Prediction execution logic with seamless fallback
if predict_btn:
    payload = {
        # 10 Numeric Features
        "Order_Hour": int(order_hour),
        "Is_Weekend": 1 if is_weekend_cb else 0,
        "Is_Festival": 1 if is_festival_cb else 0,
        "Rider_Experience_Years": float(rider_experience_years),
        "Rider_Rating": float(rider_rating),
        "Restaurant_Rating": float(restaurant_rating),
        "Order_Items": int(order_items),
        "Preparation_Time_Min": int(preparation_time_min),
        "Road_Distance_km": float(road_distance_km),
        "Number_of_Signals": int(number_of_signals),
        
        # 10 Categorical Features
        "Day_of_Week": day_of_week,
        "Weather": weather,
        "Pickup_Zone": pickup_zone,
        "Dropoff_Zone": dropoff_zone,
        "Vehicle_Type": vehicle_type,
        "Cuisine_Type": cuisine_type,
        "Restaurant_Load": restaurant_load,
        "Delivery_Distance_Category": delivery_distance_category,
        "Traffic_Level": traffic_level,
        "Delivery_Priority": delivery_priority,
    }

    try:
        api_handled = False
        response = None

        # 1. Try FastAPI microservice endpoint first
        try:
            with st.spinner("Calculating ETA via FastAPI..."):
                response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=2.0)
                if response.status_code == 200:
                    st.session_state["last_result"] = response.json()
                    st.session_state["last_error"] = None
                    api_handled = True
                else:
                    st.session_state["last_error"] = f"API Error ({response.status_code}): {response.text}"
                    st.session_state["last_result"] = None
                    api_handled = True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            api_handled = False

        # 2. If FastAPI is unreachable (e.g. Streamlit Cloud), execute seamlessly in Python
        if not api_handled and st.session_state.get("last_error") is None:
            with st.spinner("Calculating ETA via in-process XGBoost..."):
                pipeline = load_fallback_model()
                input_df = pd.DataFrame([payload])
                raw_pred = float(pipeline.predict(input_df)[0])
                estimated_minutes = round(max(0.0, raw_pred), 2)
                delay_risk = "High" if estimated_minutes > 90 else "Normal"

                record_id = insert_prediction(
                    road_distance_km=payload["Road_Distance_km"],
                    traffic_level=payload["Traffic_Level"],
                    weather=payload["Weather"],
                    vehicle_type=payload["Vehicle_Type"],
                    preparation_time_min=payload["Preparation_Time_Min"],
                    predicted_eta_mins=estimated_minutes,
                    delay_risk=delay_risk
                )

                st.session_state["last_result"] = {
                    "estimated_minutes": estimated_minutes,
                    "delay_risk": delay_risk,
                    "record_id": record_id,
                    "message": "Prediction generated and saved to database"
                }
                st.session_state["last_error"] = None

    except Exception as e:
        st.session_state["last_error"] = f"Unexpected error: {str(e)}"
        st.session_state["last_result"] = None

with col_pred:
    with st.container(key="prediction_panel_card"):
        st.markdown(
            """
            <div class="panel-header">
                <div class="panel-badge">LIVE INFERENCE</div>
                <h3 class="panel-title">⚡ Live ETA Prediction</h3>
                <p class="panel-subtitle">Real-time Machine Learning transit inference & risk classification</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        last_result = st.session_state.get("last_result")
        last_error = st.session_state.get("last_error")

        if last_error:
            if "Could not connect" in last_error:
                st.warning(last_error)
            else:
                st.error(last_error)
        elif last_result:
            eta_mins = last_result.get("estimated_minutes", 0.0)
            delay_risk = last_result.get("delay_risk", "Normal")
            record_id = last_result.get("record_id")

            st.markdown("### 📊 Prediction Results")
            metric_col, status_col = st.columns([1, 1.4])

            with metric_col:
                st.metric(label="Estimated Delivery Time", value=f"{eta_mins:.1f} mins", delta=f"Order #{record_id}")

            with status_col:
                if delay_risk == "High":
                    st.error("🚨 **High Delay Risk Detected!** Predicted ETA exceeds 90 minutes. Dispatch intervention recommended.")
                else:
                    st.success("✅ **Normal Delivery Time.** The delivery is expected on schedule without major delays.")

            st.markdown(
                f"""
                <div class="telemetry-grid">
                    <div class="telemetry-chip">
                        <span class="telemetry-label">ROUTE DISTANCE</span>
                        <span class="telemetry-val">{road_distance_km:.1f} km</span>
                    </div>
                    <div class="telemetry-chip">
                        <span class="telemetry-label">SIGNALS / TRAFFIC</span>
                        <span class="telemetry-val">{number_of_signals} • {traffic_level}</span>
                    </div>
                    <div class="telemetry-chip">
                        <span class="telemetry-label">KITCHEN PREP</span>
                        <span class="telemetry-val">{preparation_time_min} mins</span>
                    </div>
                    <div class="telemetry-chip">
                        <span class="telemetry-label">FLEET VEHICLE</span>
                        <span class="telemetry-val">{vehicle_type}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="standby-box">
                    <div class="standby-icon">🛵</div>
                    <div class="standby-title">Awaiting Dispatch Simulation</div>
                    <p class="standby-desc">Configure order parameters on the left and click <b>Predict Delivery ETA</b> to generate real-time transit calculation and delay risk classification.</p>
                    <div class="standby-pills">
                        <span class="standby-pill">⚡ Sub-10ms Inference</span>
                        <span class="standby-pill">🎯 XGBoost Regressor</span>
                        <span class="standby-pill">📦 SQLite Persisted</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# History section with seamless fallback
st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
with st.container(key="history_card"):
    hist_col1, hist_col2 = st.columns([6, 1])
    with hist_col1:
        st.subheader("📜 Recent Predictions History (SQLite)")
        st.caption("Live SQLite transaction log of verified dispatch estimations")
    with hist_col2:
        refresh_btn = st.button("🔄 Refresh History", use_container_width=True)

    history_data = None

    # Try FastAPI endpoint first
    try:
        history_res = requests.get(f"{API_BASE_URL}/history?limit=10", timeout=2.0)
        if history_res.status_code == 200:
            history_data = history_res.json().get("predictions", [])
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        history_data = None
    except Exception:
        history_data = None

    # Fallback directly to SQLite database if FastAPI is not reachable
    if history_data is None:
        try:
            history_data = get_recent_predictions(limit=10)
        except Exception as e:
            st.error(f"Error loading history from database: {str(e)}")
            history_data = []

    if history_data:
        df_history = pd.DataFrame(history_data)
        rename_map = {
            "id": "ID",
            "timestamp": "Timestamp",
            "predicted_eta_mins": "ETA (min)",
            "delay_risk": "Delay Risk",
            "road_distance_km": "Distance (km)",
            "preparation_time_min": "Prep Time (min)",
            "traffic_level": "Traffic",
            "weather": "Weather",
            "vehicle_type": "Vehicle"
        }
        display_cols = [c for c in rename_map.keys() if c in df_history.columns]
        df_display = df_history[display_cols].rename(columns=rename_map)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction records found in the database yet. Generate your first prediction above!")
