import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Smart Energy Analyzer", page_icon="⚡", layout="centered")

COST_RATE = 8       # ₹ per kWh
CO2_FACTOR = 0.82   # kg CO2 per kWh

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 10%, #1e293b 0%, #0f172a 55%, #020617 100%);
}
h1, h2, h3, p, label, .stMarkdown { color: #e2e8f0 !important; }

.title-glow {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(56,189,248,0.35);
    margin-bottom: 0;
}

.card {
    background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow:
        0 20px 40px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.08);
    transform: perspective(900px) rotateX(0.5deg);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: perspective(900px) rotateX(0deg) translateY(-3px);
    box-shadow:
        0 26px 50px rgba(0,0,0,0.55),
        inset 0 1px 0 rgba(255,255,255,0.1);
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(56,189,248,0.10), rgba(255,255,255,0.03));
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 16px;
    padding: 20px 10px;
    text-align: center;
    box-shadow:
        0 15px 30px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.08);
    transition: transform 0.25s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-4px); }
div[data-testid="stMetricValue"] {
    color: #38bdf8;
    font-size: 1.7rem;
    text-shadow: 0 0 12px rgba(56,189,248,0.5);
}

.stButton>button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6, #818cf8);
    background-size: 200% auto;
    color: white; font-weight: 700; border: none;
    border-radius: 12px; padding: 0.7rem 0; width: 100%;
    box-shadow: 0 10px 25px rgba(59,130,246,0.45);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-position: right center;
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 16px 32px rgba(59,130,246,0.6);
}
.stButton>button:active { transform: translateY(0) scale(0.99); }

input, .stSelectbox div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.3) !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open("models/cost_prediction.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_scaler():
    data = pd.read_csv("data/raw_data.csv")
    X = data[['Building Type', 'Square Footage', 'Number of Occupants',
              'Appliances Used', 'Average Temperature', 'Day of Week']]
    y = data['Energy Consumption']
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = pd.get_dummies(
        X_train, columns=['Building Type', 'Day of Week'], drop_first=True).astype(int)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler, X_train.columns.tolist()


model = load_model()
scaler, feature_cols = load_scaler()

st.markdown("<div class='title-glow'>⚡ Smart Energy Analyzer</div>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#94a3b8; margin-top:6px;'>Predict building energy consumption, cost, and carbon footprint</p>",
    unsafe_allow_html=True
)
st.write("")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        b_type = st.selectbox("🏢 Building Type", ["Residential", "Commercial", "Industrial"])
        area = st.number_input("📐 Building Area (sq ft)", min_value=100, value=5000)
        occupants = st.number_input("👥 Occupants", min_value=1, value=120)
    with col2:
        temp = st.number_input("🌡️ Average Temperature (°C)", value=31.0)
        appliances = st.number_input("🔌 Appliances", min_value=1, value=25)
        day_type = st.selectbox("📅 Day Type", ["Weekday", "Weekend"])
    st.markdown('</div>', unsafe_allow_html=True)

    predict = st.button("PREDICT ⚡")

if predict:
    row = {c: 0 for c in feature_cols}
    row['Square Footage'] = area
    row['Number of Occupants'] = occupants
    row['Appliances Used'] = appliances
    row['Average Temperature'] = temp
    if b_type == "Industrial":
        row['Building Type_Industrial'] = 1
    elif b_type == "Residential":
        row['Building Type_Residential'] = 1
    if day_type == "Weekend":
        row['Day of Week_Weekend'] = 1

    input_df = pd.DataFrame([row])[feature_cols]
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_cols)
    pred = model.predict(input_scaled)[0]

    cost = pred * COST_RATE
    co2 = pred * CO2_FACTOR

    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("⚡ Predicted Consumption", f"{pred:,.0f} kWh")
    c2.metric("💰 Estimated Cost", f"₹{cost:,.0f}")
    c3.metric("🌱 Estimated CO₂", f"{co2:,.0f} kg")