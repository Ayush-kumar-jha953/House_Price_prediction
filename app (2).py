
import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="centered")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
  html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
  h1, h2, h3 { font-family: "Syne", sans-serif !important; }
  .result-box {
      padding: 2rem; border-radius: 16px; text-align: center;
      font-family: Syne, sans-serif; font-size: 1.6rem;
      font-weight: 700; margin-top: 1.5rem;
      background: linear-gradient(135deg, #1a1a2e, #16213e);
      border: 2px solid #7c3aed; color: white;
      box-shadow: 0 0 40px rgba(124,58,237,0.3);
  }
  .price { font-size: 2.2rem; color: #a78bfa; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

bundle = load_model()
model  = bundle["model"]

st.markdown("# 🏠 California House Price Predictor")
st.markdown("##### Adjust the property details below to get an instant price estimate.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    longitude          = st.slider("Longitude",          -124.18, -114.49, -119.59, step=0.01)
    latitude           = st.slider("Latitude",             32.56,   41.92,   35.64, step=0.01)
    housing_median_age = st.slider("Housing Median Age",       1,      52,      29)
    total_rooms        = st.slider("Total Rooms",              6,   30450,    2600)

with col2:
    total_bedrooms = st.slider("Total Bedrooms",   2,  5419,  530)
    population     = st.slider("Population",       5, 11935, 1403)
    households     = st.slider("Households",       2,  4930,  490)
    median_income  = st.slider("Median Income", 0.50, 15.00,  3.81, step=0.01)

st.divider()

if st.button("🔍 Predict House Price", use_container_width=True):
    input_data = np.array([[
        longitude, latitude, housing_median_age,
        total_rooms, total_bedrooms, population,
        households, median_income
    ]])

    predicted_price = model.predict(input_data)[0]
    predicted_price = max(0, predicted_price)
    low  = predicted_price * 0.90
    high = predicted_price * 1.10

    st.markdown(f"""
    <div class="result-box">
        🏡 Estimated House Price<br>
        <span class="price">${predicted_price:,.0f}</span><br>
        <span style="font-size:0.9rem;font-weight:400;color:#c4b5fd">
            Likely range: ${low:,.0f} — ${high:,.0f}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Input Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Age",    f"{housing_median_age} yrs")
    c2.metric("Total Rooms",   f"{total_rooms:,}")
    c3.metric("Households",    f"{households:,}")
    c4.metric("Median Income", f"${median_income:.2f}k")

st.divider()
st.caption("Built with scikit-learn & Streamlit · California Housing Dataset · For educational purposes only")
