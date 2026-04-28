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

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

/* ---- Root palette ---- */
:root {
    --neon-orange : #FF6B00;
    --neon-pink   : #FF2D78;
    --neon-cyan   : #00E5FF;
    --neon-lime   : #AAFF00;
    --bg-dark     : #0D0D0D;
    --card-bg     : #161616;
    --text-main   : #F5F5F5;
    --text-muted  : #888;
    --border      : #2A2A2A;
}

/* ---- Global reset ---- */
html, body, [class*="css"] {
    background-color: var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ---- Hero banner ---- */
.hero {
    background: linear-gradient(135deg, #FF6B00 0%, #FF2D78 50%, #7B2FFF 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(255,107,0,0.35), 0 0 120px rgba(255,45,120,0.2);
}
.hero::before {
    content: "";
    position: absolute;
    top: -40%; left: -30%;
    width: 160%; height: 200%;
    background: radial-gradient(ellipse at 50% 0%, rgba(255,255,255,0.12) 0%, transparent 70%);
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    color: #fff !important;
    margin: 0 !important;
    text-shadow: 0 2px 20px rgba(0,0,0,0.4);
}
.hero p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 1.05rem;
    margin-top: 0.5rem !important;
}

/* ---- Section cards ---- */
.section-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 30px rgba(0,0,0,0.4);
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--neon-cyan) !important;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---- Streamlit widgets tweaks ---- */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--neon-orange), var(--neon-pink)) !important;
}
.stSlider [data-testid="stThumbValue"] {
    color: var(--neon-lime) !important;
    font-weight: 700 !important;
}
div[data-baseweb="select"] > div {
    background: #1E1E1E !important;
    border-color: #333 !important;
    color: var(--text-main) !important;
    border-radius: 10px !important;
}
.stNumberInput input {
    background: #1E1E1E !important;
    color: var(--text-main) !important;
    border-color: #333 !important;
    border-radius: 10px !important;
}

/* ---- Predict button ---- */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #FF6B00, #FF2D78, #7B2FFF) !important;
    background-size: 200% !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 0 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 25px rgba(255,107,0,0.5) !important;
    text-transform: uppercase !important;
}
div.stButton > button:hover {
    background-position: right !important;
    box-shadow: 0 0 45px rgba(255,45,120,0.7) !important;
    transform: translateY(-2px) !important;
}

/* ---- Result box ---- */
.result-box {
    background: linear-gradient(135deg, #0D0D0D, #1a1a1a);
    border: 2px solid var(--neon-lime);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(170,255,0,0.25), inset 0 0 60px rgba(170,255,0,0.05);
    margin-top: 1.5rem;
    animation: popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
@keyframes popIn {
    from { transform: scale(0.8); opacity: 0; }
    to   { transform: scale(1);   opacity: 1; }
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--neon-lime) !important;
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #fff !important;
    line-height: 1;
    text-shadow: 0 0 30px rgba(170,255,0,0.6);
}
.result-unit {
    font-size: 1.2rem;
    color: var(--text-muted) !important;
    margin-top: 0.3rem;
}

/* ---- Tier badge ---- */
.tier-badge {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 100px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 1rem;
}
.tier-fast   { background: rgba(0,229,255,0.15); color: #00E5FF; border: 1px solid #00E5FF; }
.tier-normal { background: rgba(170,255,0,0.15); color: #AAFF00; border: 1px solid #AAFF00; }
.tier-slow   { background: rgba(255,107,0,0.15); color: #FF6B00; border: 1px solid #FF6B00; }

/* ---- Feature importance bar ---- */
.imp-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.imp-label { color: var(--text-muted); font-size: 0.78rem; width: 180px; flex-shrink: 0; }
.imp-bar-wrap { flex: 1; background: #222; border-radius: 4px; height: 8px; }
.imp-bar { height: 8px; border-radius: 4px; }

/* ---- Footer ---- */
.footer { text-align: center; color: var(--text-muted); font-size: 0.78rem; margin-top: 3rem; padding: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

model = load_model()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛵 Delivery Time Predictor</h1>
    <p>Powered by Random Forest · 100 Trees · Real-time inference</p>
</div>
""", unsafe_allow_html=True)

# ── Section 1 · Route Details ──────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📍 Route Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    distance = st.slider("Distance (km)", min_value=0.5, max_value=50.0, value=10.0, step=0.5)
with col2:
    prep_time = st.slider("Preparation Time (min)", min_value=1, max_value=60, value=15, step=1)

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2 · Courier & Vehicle ─────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🏍️ Courier & Vehicle</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    experience = st.number_input("Courier Experience (yrs)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
with col4:
    vehicle = st.selectbox("Vehicle Type", ["Bike", "Car", "Scooter"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3 · Conditions ────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌤️ Conditions</div>', unsafe_allow_html=True)

col5, col6, col7 = st.columns(3)
with col5:
    weather = st.selectbox("Weather", ["Clear", "Foggy", "Rainy", "Snowy", "Windy"])
with col6:
    traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
with col7:
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("⚡ Predict Delivery Time"):

    # Build feature vector (order must match training)
    features = {
        "Distance_km"           : distance,
        "Preparation_Time_min"  : prep_time,
        "Courier_Experience_yrs": experience,
        "Weather_Clear"         : 1 if weather == "Clear"   else 0,
        "Weather_Foggy"         : 1 if weather == "Foggy"   else 0,
        "Weather_Rainy"         : 1 if weather == "Rainy"   else 0,
        "Weather_Snowy"         : 1 if weather == "Snowy"   else 0,
        "Weather_Windy"         : 1 if weather == "Windy"   else 0,
        "Traffic_Level_High"    : 1 if traffic == "High"    else 0,
        "Traffic_Level_Low"     : 1 if traffic == "Low"     else 0,
        "Traffic_Level_Medium"  : 1 if traffic == "Medium"  else 0,
        "Time_of_Day_Afternoon" : 1 if time_of_day == "Afternoon" else 0,
        "Time_of_Day_Evening"   : 1 if time_of_day == "Evening"   else 0,
        "Time_of_Day_Morning"   : 1 if time_of_day == "Morning"   else 0,
        "Time_of_Day_Night"     : 1 if time_of_day == "Night"     else 0,
        "Vehicle_Type_Bike"     : 1 if vehicle == "Bike"    else 0,
        "Vehicle_Type_Car"      : 1 if vehicle == "Car"     else 0,
        "Vehicle_Type_Scooter"  : 1 if vehicle == "Scooter" else 0,
    }

    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]

    # Tier classification
    if prediction < 25:
        tier_class, tier_label, tier_emoji = "tier-fast",   "⚡ Express Delivery", "⚡"
    elif prediction < 45:
        tier_class, tier_label, tier_emoji = "tier-normal", "✅ On-Time Delivery", "✅"
    else:
        tier_class, tier_label, tier_emoji = "tier-slow",   "⏳ Delayed Delivery", "⏳"

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Delivery Time</div>
        <div class="result-value">{prediction:.1f}</div>
        <div class="result-unit">minutes</div>
        <div class="tier-badge {tier_class}">{tier_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature importance mini-chart ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Top Feature Importances</div>', unsafe_allow_html=True)

    importances = model.feature_importances_
    feat_names  = list(model.feature_names_in_)
    fi = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:7]
    max_imp = fi[0][1]

    colors = ["#FF6B00","#FF2D78","#7B2FFF","#00E5FF","#AAFF00","#FFD600","#FF6B00"]
    bars_html = ""
    for i, (name, imp) in enumerate(fi):
        pct = imp / max_imp * 100
        bars_html += f"""
        <div class="imp-row">
            <div class="imp-label">{name.replace('_', ' ')}</div>
            <div class="imp-bar-wrap">
                <div class="imp-bar" style="width:{pct:.1f}%; background:{colors[i % len(colors)]};"></div>
            </div>
            <span style="font-size:0.78rem;color:#aaa;width:42px;text-align:right">{imp:.3f}</span>
        </div>"""

    st.markdown(bars_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Random Forest Regressor · 100 estimators · max_depth=20 · sklearn 1.6.1
</div>
""", unsafe_allow_html=True)