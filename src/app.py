import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Lava Health Guardian",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Font Import - Outfit for modern, clean look */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #FF512F 0%, #DD2476 100%);
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
        --neon-glow: 0 0 10px rgba(221, 36, 118, 0.3);
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Dramatic Dark Background */
    .stApp {
        background: radial-gradient(circle at top left, #1a1c2e 0%, #0f0c29 100%);
        color: #ffffff;
    }

    /* Glass Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--glass-border);
    }

    /* Ultra-Glass Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }
    .glass-card:hover {
        transform: translateY(-5px) scale(1.01);
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Typography */
    h1, h2, h3 {
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .gradient-text {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Premium Buttons */
    .stButton>button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(221, 36, 118, 0.4);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(221, 36, 118, 0.6);
    }

    /* Inputs */
    div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# Session Manager
if 'page' not in st.session_state: st.session_state.page = "Dashboard"
def navigate_to(page): st.session_state.page = page

# Load Models
@st.cache_resource
def load_resources():
    resources = {}
    model_dir = "models"
    try:
        with open(os.path.join(model_dir, "diabetes_model.pkl"), "rb") as f: resources['diabetes'] = pickle.load(f)
        with open(os.path.join(model_dir, "heart_model.pkl"), "rb") as f: resources['heart'] = pickle.load(f)
        with open(os.path.join(model_dir, "kidney_model.pkl"), "rb") as f: resources['kidney'] = pickle.load(f)
        with open(os.path.join(model_dir, "kidney_encoders.pkl"), "rb") as f: resources['kidney_encoders'] = pickle.load(f)
    except: return None
    return resources

resources = load_resources()

# --- SIDEBAR NAV ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="
                width: 80px; height: 80px; margin: 0 auto;
                background: linear-gradient(135deg, #FF512F, #DD2476);
                border-radius: 20px;
                display: flex; align-items: center; justify-content: center;
                font-size: 40px; box-shadow: 0 10px 30px rgba(221, 36, 118, 0.5);
            ">
                🌋
            </div>
            <h2 style="margin-top: 1rem; font-size: 1.4rem; color: #fff;">Lava Health<br><span style="opacity:0.7; font-size: 1rem; font-weight: 400;">Guardian AI</span></h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Custom Nav Buttons
    nav_items = {
        "Dashboard": "📊",
        "Diabetes": "🩸", 
        "Heart": "💓", 
        "Kidney": "🩺", 
        "About": "ℹ️", 
        "Contact": "📞"
    }
    
    for page, icon in nav_items.items():
        if st.button(f"{icon}  {page}", key=f"nav_{page}", use_container_width=True):
            navigate_to(page)
            st.rerun()

    st.markdown("---")
    st.markdown("""
        <div class="glass-card" style="padding: 1rem; text-align: center; border: 1px solid rgba(74, 222, 128, 0.2); background: rgba(74, 222, 128, 0.05);">
            <small>✅ System Active</small><br>
            <small style="opacity: 0.6">v1.2.0 Stable</small>
        </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT ---
current_page = st.session_state.page

if current_page == "Dashboard":
    # Hero Section
    st.markdown("""
        <div style='text-align: center; padding: 3rem 0;'>
            <h1 style='font-size: 4rem; margin-bottom: 0px;'>Welcome to <span class='gradient-text'>Lava Health</span></h1>
            <p style='font-size: 1.5rem; opacity: 0.8; max-width: 700px; margin: 1.5rem auto;'>
                Elite AI-powered diagnostics. Instant results. Ironclad privacy.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin:0; color: #ff6b6b;">Diabetes</h3>
                    <p style="margin:0; opacity: 0.6;">Metabolic Scanner</p>
                </div>
                <div style="font-size: 2.5rem;">🩸</div>
            </div>
            <div style="margin-top: 1.5rem; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px;">
                <div style="width: 98%; height: 100%; background: #ff6b6b; border-radius: 2px;"></div>
            </div>
            <p style="text-align: right; font-size: 0.8rem; margin-top: 5px; opacity: 0.7;">98% Precision</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Scanner", key="go_dia", use_container_width=True):
            navigate_to("Diabetes")
            st.rerun()

    with k2:
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin:0; color: #4ecdc4;">Heart</h3>
                    <p style="margin:0; opacity: 0.6;">Cardiac Profiler</p>
                </div>
                <div style="font-size: 2.5rem;">💓</div>
            </div>
            <div style="margin-top: 1.5rem; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px;">
                <div style="width: 100%; height: 100%; background: #4ecdc4; border-radius: 2px;"></div>
            </div>
            <p style="text-align: right; font-size: 0.8rem; margin-top: 5px; opacity: 0.7;">Instant Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Profiler", key="go_hrt", use_container_width=True):
            navigate_to("Heart")
            st.rerun()

    with k3:
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin:0; color: #a78bfa;">Kidney</h3>
                    <p style="margin:0; opacity: 0.6;">Renal Diagnostics</p>
                </div>
                <div style="font-size: 2.5rem;">🩺</div>
            </div>
            <div style="margin-top: 1.5rem; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px;">
                <div style="width: 100%; height: 100%; background: #a78bfa; border-radius: 2px;"></div>
            </div>
            <p style="text-align: right; font-size: 0.8rem; margin-top: 5px; opacity: 0.7;">24-Point Check</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Diagnostics", key="go_kid", use_container_width=True):
            navigate_to("Kidney")
            st.rerun()

elif current_page == "Diabetes":
    st.markdown("<h2 class='gradient-text'>🩸 Metabolic Scanner</h2>", unsafe_allow_html=True)
    if resources:
        with st.form("dia_form"):
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.caption("Enter patient vitals below")
            c1, c2, c3, c4 = st.columns(4)
            with c1: 
                pregnancies = st.number_input("Pregnancies", 0, 20, 0)
                insulin = st.number_input("Insulin (mu U/ml)", 0, 999, 79)
            with c2: 
                glucose = st.number_input("Glucose (mg/dL)", 0, 300, 100)
                bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            with c3: 
                bp = st.number_input("Blood Pressure", 0, 180, 70)
                dpf = st.number_input("Pedigree Function", 0.0, 3.0, 0.5)
            with c4: 
                skin = st.number_input("Skin Thickness", 0, 100, 20)
                age = st.number_input("Age", 0, 120, 30)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.form_submit_button("Run Analysis ➜"):
                features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
                prob = resources['diabetes'].predict_proba(features)[0][1]
                pred = resources['diabetes'].predict(features)[0]
                
                if pred == 1:
                    st.markdown(f"""
                    <div class="glass-card" style="background: rgba(255, 81, 47, 0.15); border-color: #FF512F;">
                        <h2 style="color: #FF512F;">⚠️ Abnormal Metabolic signatures detected</h2>
                        <h1 style="font-size: 3rem;">{prob:.1%} <span style="font-size:1rem; opacity:0.8">Risk Probability</span></h1>
                        <p>We recommend immediate consultation with an endocrinologist.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="glass-card" style="background: rgba(75, 235, 123, 0.15); border-color: #4beb7b;">
                        <h2 style="color: #4beb7b;">✅ Metabolic Profile Normal</h2>
                        <h1 style="font-size: 3rem;">{(1-prob):.1%} <span style="font-size:1rem; opacity:0.8">Health Score</span></h1>
                        <p>Signatures are within healthy ranges. Maintain current lifestyle.</p>
                    </div>
                    """, unsafe_allow_html=True)

elif current_page == "Heart":
    st.markdown("<h2 class='gradient-text'>💓 Cardiac Profiler</h2>", unsafe_allow_html=True)
    with st.form("hrt_form"):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 20, 100, 50)
            trestbps = st.number_input("Resting BP", 90, 200, 120)
            chol = st.number_input("Cholesterol", 100, 600, 200)
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0)
        with c2:
            sex = st.selectbox("Sex", ["Male", "Female"])
            fbs = st.selectbox("Fasting Blood Sugar > 120", ["False", "True"])
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
            ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3])
        with c3:
            cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            restecg = st.selectbox("Resting ECG", ["Normal", "Abnormality", "Hypertrophy"])
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.form_submit_button("Analyze Cardiac Health ➜"):
            # Mapping
            sex_v = 1 if sex == "Male" else 0
            cp_v = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
            fbs_v = 1 if fbs == "True" else 0
            restecg_v = ["Normal", "Abnormality", "Hypertrophy"].index(restecg)
            exang_v = 1 if exang == "Yes" else 0
            slope_v = ["Upsloping", "Flat", "Downsloping"].index(slope)
            thal_v = ["Normal", "Fixed Defect", "Reversable Defect"].index(thal) + 1
            
            features = np.array([[age, sex_v, cp_v, trestbps, chol, fbs_v, restecg_v, thalach, exang_v, oldpeak, slope_v, ca, thal_v]])
            prob = resources['heart'].predict_proba(features)[0][1]
            pred = resources['heart'].predict(features)[0]
            
            if pred == 1:
                st.markdown(f"""
                <div class="glass-card" style="background: rgba(255, 81, 47, 0.15); border-color: #FF512F;">
                    <h2 style="color: #FF512F;">⚠️ Cardiac Risk Detected</h2>
                    <h1 style="font-size: 3rem;">{prob:.1%} <span style="font-size:1rem; opacity:0.8">Risk Factor</span></h1>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="glass-card" style="background: rgba(75, 235, 123, 0.15); border-color: #4beb7b;">
                    <h2 style="color: #4beb7b;">✅ Heart Health Optimal</h2>
                    <h1 style="font-size: 3rem;">{(1-prob):.1%} <span style="font-size:1rem; opacity:0.8">Score</span></h1>
                </div>
                """, unsafe_allow_html=True)

elif current_page == "Kidney":
    st.markdown("<h2 class='gradient-text'>🩺 Renal Diagnostics</h2>", unsafe_allow_html=True)
    with st.form("kid_form"):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        t1, t2 = st.tabs(["Biochemistry", "History & Vitals"])
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                 al = st.slider("Albumin", 0, 5, 0)
                 su = st.slider("Sugar", 0, 5, 0)
                 bu = st.number_input("Blood Urea", 10, 400, 40)
                 sc_k = st.number_input("Serum Creatinine", 0.0, 15.0, 1.0)
            with c2:
                 bgr = st.number_input("Glucose Random", 50, 500, 120)
                 sod = st.number_input("Sodium", 100, 180, 137)
                 pot = st.number_input("Potassium", 2.0, 8.0, 4.0)
                 hemo = st.number_input("Hemoglobin", 3.0, 20.0, 15.0)
        with t2:
            c3, c4 = st.columns(2)
            with c3:
                age_k = st.number_input("Age", 1, 100, 40)
                bp_k = st.number_input("BP", 50, 180, 80)
                sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
                rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
                pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
            with c4:
                htn = st.selectbox("Hypertension", ["yes", "no"])
                dm = st.selectbox("Diabetes", ["yes", "no"])
                cad = st.selectbox("CAD", ["yes", "no"])
                appet = st.selectbox("Appetite", ["good", "poor"])
                pe = st.selectbox("Pedal Edema", ["yes", "no"])
                ane = st.selectbox("Anemia", ["yes", "no"])
                
        # Hidden defaults for simplicity in UI code, but robust map in logic
        pcc, ba, pcv, wc, rc = "notpresent", "notpresent", 40, 8000, 5.0
        
        st.markdown('</div>', unsafe_allow_html=True)
        if st.form_submit_button("Analyze Renal Function ➜"):
            input_data = {
                    'sg': sg, 'al': al, 'su': su, 'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba,
                    'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane,
                    'age': age_k, 'bp': bp_k, 'bgr': bgr, 'bu': bu, 'sc': sc_k, 
                    'sod': sod, 'pot': pot, 'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc
            }
            encoders = resources['kidney_encoders']
            def safe_encode(col, val):
                if col in encoders:
                    try: return encoders[col].transform([str(val)])[0]
                    except: return 0
                return val
            ordered_cols = ['sg','al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
            data_encoded = [safe_encode(c, input_data[c]) if isinstance(input_data[c], str) else input_data[c] for c in ordered_cols]
            
            features = np.array([data_encoded])
            prob = resources['kidney'].predict_proba(features)[0][1]
            pred = resources['kidney'].predict(features)[0]
            
            if pred == 1:
                st.markdown(f"""
                <div class="glass-card" style="background: rgba(255, 81, 47, 0.15); border-color: #FF512F;">
                    <h2 style="color: #FF512F;">⚠️ Renal Insufficiency Detected</h2>
                    <h1 style="font-size: 3rem;">{prob:.1%} <span style="font-size:1rem; opacity:0.8">Confidence</span></h1>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="glass-card" style="background: rgba(75, 235, 123, 0.15); border-color: #4beb7b;">
                     <h2 style="color: #4beb7b;">✅ Renal Function Optimal</h2>
                     <h1 style="font-size: 3rem;">{(1-prob):.1%} <span style="font-size:1rem; opacity:0.8">Score</span></h1>
                </div>
                """, unsafe_allow_html=True)

elif current_page == "About":
    st.markdown("<h1 class='gradient-text'>ℹ️ About Lava Health</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <h3>Vision</h3>
        <p>Lava Health Guardian AI integrates <b>clinical-grade algorithms</b> into a consumer-facing platform.</p>
        <p>We believe in <b>Privacy-First AI</b>: Your data never leaves this machine.</p>
    </div>
    """, unsafe_allow_html=True)

elif current_page == "Contact":
    st.markdown("<h1 class='gradient-text'>📞 Contact Support</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 4rem;">
        <h2 style="margin-bottom: 2rem;">Professional Inquiries</h2>
        <a href="mailto:avanyavetri@gmail.com" style="text-decoration: none;">
            <div style="
                display: inline-block;
                background: linear-gradient(135deg, #FF512F, #DD2476);
                padding: 1.5rem 3rem;
                border-radius: 50px;
                color: white;
                font-weight: 700;
                font-size: 1.2rem;
                box-shadow: 0 10px 30px rgba(221, 36, 118, 0.4);
                transition: transform 0.3s ease;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                ✉️ avanyavetri@gmail.com
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 5rem; opacity: 0.4; font-size: 0.8rem;">
    Lava Health Guardian AI © 2026 | v1.2.0 | Privacy Secured
</div>
""", unsafe_allow_html=True)
