import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Delivery Time Predictor",
    page_icon="🛵",
    layout="centered",
)

# ── Custom CSS (chamki / neon theme) ──────────────────────────────────────────
st.markdown("""
<style>
/* ---- background gradient ---- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }

/* ---- neon title ---- */
.neon-title {
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    color: #fff;
    text-shadow:
        0 0 7px #ff00de,
        0 0 20px #ff00de,
        0 0 40px #ff00de,
        0 0 80px #ff00de;
    letter-spacing: 3px;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center;
    color: #c9b1ff;
    font-size: 1rem;
    margin-bottom: 2rem;
    letter-spacing: 1px;
}

/* ---- glowing cards ---- */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,100,255,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 0 18px rgba(255,0,220,0.15);
}
.card-title {
    color: #ff79c6;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}

/* ---- result box ---- */
.result-box {
    background: linear-gradient(90deg, #f093fb, #f5576c);
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(240,93,251,0.6);
    animation: pulse 2s infinite;
}
.result-box h2 {
    color: white;
    font-size: 1.2rem;
    margin: 0 0 0.4rem 0;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.result-box h1 {
    color: white;
    font-size: 3.5rem;
    font-weight: 900;
    margin: 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.result-box p { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 0.95rem; }

@keyframes pulse {
    0%   { box-shadow: 0 0 30px rgba(240,93,251,0.5); }
    50%  { box-shadow: 0 0 60px rgba(240,93,251,0.9); }
    100% { box-shadow: 0 0 30px rgba(240,93,251,0.5); }
}

/* ---- labels / selects ---- */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #e0d4ff !important;
    font-weight: 600 !important;
}
.stSelectbox > div > div { background: rgba(255,255,255,0.08) !important; color: white !important; }

/* ---- predict button ---- */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    color: white;
    font-size: 1.1rem;
    font-weight: 800;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    box-shadow: 0 0 25px rgba(240,93,251,0.5);
    transition: all 0.3s ease;
    cursor: pointer;
}
.stButton > button:hover {
    box-shadow: 0 0 50px rgba(240,93,251,0.9);
    transform: translateY(-2px);
}

/* ---- divider ---- */
hr { border-color: rgba(255,100,255,0.2); }

/* ---- slider track ---- */
[data-testid="stSlider"] .st-bj { background: #f093fb !important; }

</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("C:/Users/amand/Downloads/rf_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="neon-title">🛵 DELIVERY TIME</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">✨ Powered by Random Forest · Predict your ETA instantly ✨</div>', unsafe_allow_html=True)

if not model_loaded:
    st.error(f"❌ Could not load model: {load_error}")
    st.stop()

# ── Input Cards ────────────────────────────────────────────────────────────────

# Card 1: Route Info
st.markdown('<div class="card"><div class="card-title">📍 Route Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    distance = st.slider("Distance (km)", min_value=0.5, max_value=50.0, value=8.0, step=0.5)
with col2:
    prep_time = st.slider("Preparation Time (min)", min_value=1, max_value=60, value=15)
st.markdown('</div>', unsafe_allow_html=True)

# Card 2: Courier & Vehicle
st.markdown('<div class="card"><div class="card-title">🏍️ Courier & Vehicle</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    courier_exp = st.slider("Courier Experience (yrs)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
with col4:
    vehicle = st.selectbox("Vehicle Type", ["Bike", "Car", "Scooter"])
st.markdown('</div>', unsafe_allow_html=True)

# Card 3: Conditions
st.markdown('<div class="card"><div class="card-title">🌦️ Conditions</div>', unsafe_allow_html=True)
col5, col6, col7 = st.columns(3)
with col5:
    weather = st.selectbox("Weather", ["Clear", "Foggy", "Rainy", "Snowy", "Windy"])
with col6:
    traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
with col7:
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Build feature vector ────────────────────────────────────────────────────────
def build_features():
    features = {
        'Distance_km':              distance,
        'Preparation_Time_min':     prep_time,
        'Courier_Experience_yrs':   courier_exp,
        'Weather_Clear':            int(weather == "Clear"),
        'Weather_Foggy':            int(weather == "Foggy"),
        'Weather_Rainy':            int(weather == "Rainy"),
        'Weather_Snowy':            int(weather == "Snowy"),
        'Weather_Windy':            int(weather == "Windy"),
        'Traffic_Level_High':       int(traffic == "High"),
        'Traffic_Level_Low':        int(traffic == "Low"),
        'Traffic_Level_Medium':     int(traffic == "Medium"),
        'Time_of_Day_Afternoon':    int(time_of_day == "Afternoon"),
        'Time_of_Day_Evening':      int(time_of_day == "Evening"),
        'Time_of_Day_Morning':      int(time_of_day == "Morning"),
        'Time_of_Day_Night':        int(time_of_day == "Night"),
        'Vehicle_Type_Bike':        int(vehicle == "Bike"),
        'Vehicle_Type_Car':         int(vehicle == "Car"),
        'Vehicle_Type_Scooter':     int(vehicle == "Scooter"),
    }
    return pd.DataFrame([features])

# ── Predict ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("⚡ PREDICT DELIVERY TIME")

if predict_btn:
    X = build_features()
    prediction = model.predict(X)[0]

    # emoji badge based on speed
    if prediction < 20:
        badge = "🚀 Lightning Fast!"
    elif prediction < 40:
        badge = "⚡ On Time!"
    elif prediction < 60:
        badge = "🕐 Moderate Delay"
    else:
        badge = "⚠️ Expect a Wait"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-box">
        <h2>⏱ Estimated Delivery Time</h2>
        <h1>{prediction:.1f} <span style='font-size:1.8rem'>min</span></h1>
        <p>{badge}</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance mini-table
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 Top Feature Importances"):
        fi = pd.Series(model.feature_importances_, index=model.feature_names_in_)
        fi = fi.sort_values(ascending=False).head(8).reset_index()
        fi.columns = ["Feature", "Importance"]
        fi["Importance"] = fi["Importance"].map(lambda x: f"{x:.4f}")
        st.dataframe(fi, use_container_width=True, hide_index=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#9b7fe8;font-size:0.8rem;">🌟 Built with Streamlit · Random Forest Regressor · 100 Estimators · 18 Features</p>',
    unsafe_allow_html=True
)