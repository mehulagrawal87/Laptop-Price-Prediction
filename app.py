import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set page configurations
st.set_page_config(
    page_title="Laptop Price Prediction | Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Load model and dataframe
try:
    import sklearn
    df = pickle.load(open("df.pkl", "rb"))
    pipe = pickle.load(open("pipe.pkl", "rb"))
except (ModuleNotFoundError, AttributeError, KeyError) as e:
    # Automatically retrain the model in the current environment if loading fails
    st.info("🔄 Optimizing the machine learning model for your current environment... This will only take a moment.")
    try:
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
        from sklearn.linear_model import Ridge
        from xgboost import XGBRegressor
        import inspect
        
        # Load the dataframe
        df = pickle.load(open("df.pkl", "rb"))
        
        X = df.drop(columns=['Price'])
        y = np.log(df['Price'])
        
        # Check compatibility for OneHotEncoder arguments across scikit-learn versions
        ohe_params = inspect.signature(OneHotEncoder).parameters
        if 'sparse_output' in ohe_params:
            ohe = OneHotEncoder(sparse_output=False, drop='first')
        else:
            ohe = OneHotEncoder(sparse=False, drop='first')
            
        step1 = ColumnTransformer(transformers=[
            ('col_tnf', ohe, [0, 1, 7, 10, 11])
        ], remainder='passthrough')
        
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=350, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)),
            ('gbdt', GradientBoostingRegressor(n_estimators=100, max_features=0.5)),
            ('xgb', XGBRegressor(n_estimators=25, learning_rate=0.3, max_depth=5))
        ]
        
        step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))
        
        pipe = Pipeline([
            ('step1', step1),
            ('step2', step2)
        ])
        
        # Fit on the entire dataset to maximize model accuracy
        pipe.fit(X, y)
        
        # Save the compatible model to disk
        pickle.dump(pipe, open("pipe.pkl", "wb"))
        st.success("✅ Model successfully optimized and saved!")
        st.rerun()
    except Exception as retrain_err:
        st.error(f"""
        ### ⚠️ System Initialization Error
        
        Failed to automatically compile the machine learning model.
        
        * **Error Details:** `{retrain_err}`
        
        **Please run the application using the project's virtual environment which contains the correct scikit-learn version:**
        
        ```bash
        .\\.venv\\Scripts\\streamlit run app.py
        ```
        
        Or install the required packages in your current environment:
        ```bash
        pip install scikit-learn xgboost pandas numpy
        ```
        """)
        st.stop()


# Inject Futuristic Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Orbitron:wght@400;600;800;900&display=swap');

/* Hide Streamlit default components */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div.stDeployButton {display: none;}

/* Main application background and typography */
.stApp {
    background: radial-gradient(circle at 50% 50%, #0f1326 0%, #04050a 100%) !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Titles and Headers */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif !important;
    color: #FFFFFF !important;
    letter-spacing: 0.5px !important;
}

/* Custom styled card containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 19, 38, 0.45) !important;
    border: 1px solid rgba(0, 242, 254, 0.08) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), 0 0 25px rgba(0, 242, 254, 0.02) !important;
    backdrop-filter: blur(16px) !important;
    margin-bottom: 24px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(0, 242, 254, 0.2) !important;
    box-shadow: 0 16px 45px rgba(0, 0, 0, 0.6), 0 0 30px rgba(0, 242, 254, 0.06) !important;
    transform: translateY(-2px);
}

/* Dropdown (Selectbox) Customization */
div[data-baseweb="select"] > div {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="select"] > div:hover, div[data-baseweb="select"] > div:focus-within {
    border-color: #00F2FE !important;
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.15) !important;
}

/* Number inputs styling */
div[data-testid="stNumberInput"] [data-testid="stNumberInputContainer"] {
    background-color: transparent !important;
    border: none !important;
}

div[data-testid="stNumberInput"] > div {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stNumberInput"] > div:focus-within {
    border-color: #00F2FE !important;
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.15) !important;
}

/* Force all inner nested divs to be transparent so the wrapper color shows through */
div[data-testid="stNumberInput"] > div div {
    background-color: transparent !important;
}

div[data-testid="stNumberInput"] input {
    background-color: transparent !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

div[data-testid="stNumberInput"] button {
    background-color: transparent !important;
    border: none !important;
    color: #FFFFFF !important;
}

div[data-testid="stNumberInput"] button:hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
}

/* Slider modifications */
div[data-testid="stSlider"] > div > div > div > div {
    background-color: #00F2FE !important;
}
div[data-testid="stSlider"] [data-testid="stSliderTickBar"] {
    color: #64748B !important;
}

/* Custom styled action button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #9B51E0 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    letter-spacing: 2px !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 242, 254, 0.25) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase;
    margin-top: 15px;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(0, 242, 254, 0.45), 0 0 15px rgba(155, 81, 224, 0.3) !important;
    letter-spacing: 2.5px !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* Custom app header */
.app-header {
    text-align: center;
    padding: 30px 0;
    margin-bottom: 30px;
    background: linear-gradient(180deg, rgba(15, 19, 38, 0.6) 0%, rgba(5, 7, 10, 0) 100%);
    border-bottom: 1px solid rgba(0, 242, 254, 0.05);
}

.title-glow {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 3rem;
    text-transform: uppercase;
    letter-spacing: 4px;
    background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 33%, #9B51E0 66%, #00FF87 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(0, 242, 254, 0.2);
    margin: 0;
}

.subtitle {
    color: #94A3B8;
    font-size: 1.1rem;
    margin-top: 10px;
    letter-spacing: 2px;
    font-weight: 300;
    text-transform: uppercase;
}

/* Card titles with custom colored accents */
.section-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title.blue { color: #00F2FE; }
.section-title.purple { color: #9B51E0; }
.section-title.green { color: #00FF87; }

/* Pulse animation for waiting svg */
@keyframes pulse {
    0% { transform: scale(1); opacity: 0.7; }
    50% { transform: scale(1.05); opacity: 1; }
    100% { transform: scale(1); opacity: 0.7; }
}
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="app-header">
    <h1 class="title-glow">🎮 LAPTOP PRICE PREDICTION</h1>
    <p class="subtitle">Next-Gen Laptop Price Forecasting Engine</p>
</div>
""", unsafe_allow_html=True)

# Main Grid Layout
col_left, col_right = st.columns([7, 5], gap="large")

# Configuration Inputs (Left Column)
with col_left:
    st.markdown("<h3 style='margin-bottom: 20px; font-family: \"Orbitron\", sans-serif; font-size: 1.6rem;'>🎛️ CONFIGURATOR</h3>", unsafe_allow_html=True)
    
    # Section 1: Brand & platform
    with st.container(border=True):
        st.markdown('<div class="section-title blue">🌐 Identity & Portability</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            company = st.selectbox("Brand / Manufacturer", sorted(df["Company"].unique()))
        with c2:
            type_name = st.selectbox("Form Factor / Type", sorted(df["TypeName"].unique()))
        with c3:
            os_val = st.selectbox("Operating System", sorted(df["os"].unique()))
            
        c4, c5 = st.columns(2)
        with c4:
            screen_size = st.selectbox("Screen Size (Inches)", [10.1, 11.6, 12.0, 12.3, 12.5, 13.3, 13.5, 13.9, 14.0, 15.0, 15.4, 15.6, 17.0, 17.3, 18.4], index=11)
        with c5:
            weight = st.number_input("Laptop Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    # Section 2: Core Hardware
    with st.container(border=True):
        st.markdown('<div class="section-title purple">🧠 Processing & Memory</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            cpu = st.selectbox("Processor (CPU Brand)", sorted(df["Cpu brand"].unique()))
        with c2:
            gpu = st.selectbox("Graphics (GPU Brand)", sorted(df["Gpu brand"].unique()))
        with c3:
            ram = st.selectbox("System Memory (RAM GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3) # default 8GB

    # Section 3: Display, Portability & Storage
    with st.container(border=True):
        st.markdown('<div class="section-title green">🖥️ Storage & Display Fidelity</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            resolution = st.selectbox("Screen Resolution", [
                "1920x1080", "1366x768", "1600x900", "3840x2160",
                "3200x1800", "2880x1800", "2560x1600", "2560x1440", "2304x1440"
            ])
        with c2:
            ssd = st.selectbox("SSD Capacity (GB)", [0, 8, 128, 256, 512, 1024], index=3) # default 256GB
            
        c3, c4, c5 = st.columns(3)
        with c3:
            hdd = st.selectbox("HDD Capacity (GB)", [0, 128, 256, 512, 1024, 2048], index=0) # default 0GB
        with c4:
            touchscreen = st.selectbox("Touchscreen Support", ["No", "Yes"])
        with c5:
            ips = st.selectbox("IPS Panel Display", ["No", "Yes"])

# Real-time Score Engine (Calculated automatically on inputs change)
cpu_scores = {'Intel Core i7': 95, 'Intel Core i5': 75, 'AMD Processor': 80, 'Intel Core i3': 50, 'Other Intel Processor': 40}
cpu_score = cpu_scores.get(cpu, 50)

gpu_scores = {'Nvidia': 95, 'AMD': 70, 'Intel': 45}
gpu_score = gpu_scores.get(gpu, 45)

ram_scores = {64: 100, 32: 95, 24: 85, 16: 75, 12: 60, 8: 45, 6: 35, 4: 25, 2: 10}
ram_score = ram_scores.get(ram, 45)

ssd_score = 100 if ssd > 0 else 30

productivity_score = int(0.4 * ram_score + 0.5 * cpu_score + 0.1 * ssd_score)
gaming_score = int(0.55 * gpu_score + 0.3 * cpu_score + 0.15 * ram_score)

weight_factor = 100 * (1 - (weight - 0.5) / 4.5)
size_factor = 100 * (1 - (screen_size - 10.0) / 8.0)
portability_score = int(0.7 * weight_factor + 0.3 * size_factor)

# Initialize Session States for sticky predictions
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

# Forecast Control Panel (Right Column)
with col_right:
    st.markdown("<h3 style='margin-bottom: 20px; font-family: \"Orbitron\", sans-serif; font-size: 1.6rem;'>⚡ AI PRICE FORECASTER</h3>", unsafe_allow_html=True)
    
    # Specs Scorer container
    with st.container(border=True):
        st.markdown('<div class="section-title blue" style="margin-bottom: 15px;">📊 Performance Indices</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 600; color: #E2E8F0; font-size: 13px;">🚀 Productivity & Multitasking Power</span>
                <span style="font-weight: bold; color: #00F2FE; font-size: 13px;">{productivity_score}%</span>
            </div>
            <div style="background-color: rgba(255,255,255,0.03); border-radius: 10px; height: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.02);">
                <div style="background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%); width: {productivity_score}%; height: 100%; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 242, 254, 0.3);"></div>
            </div>
        </div>
        
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 600; color: #E2E8F0; font-size: 13px;">🎮 Gaming & Graphics Index</span>
                <span style="font-weight: bold; color: #9B51E0; font-size: 13px;">{gaming_score}%</span>
            </div>
            <div style="background-color: rgba(255,255,255,0.03); border-radius: 10px; height: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.02);">
                <div style="background: linear-gradient(90deg, #4FACFE 0%, #9B51E0 100%); width: {gaming_score}%; height: 100%; border-radius: 10px; box-shadow: 0 0 10px rgba(155, 81, 224, 0.3);"></div>
            </div>
        </div>
        
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 600; color: #E2E8F0; font-size: 13px;">✈️ Ultra-Portability Index</span>
                <span style="font-weight: bold; color: #00FF87; font-size: 13px;">{portability_score}%</span>
            </div>
            <div style="background-color: rgba(255,255,255,0.03); border-radius: 10px; height: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.02);">
                <div style="background: linear-gradient(90deg, #00FF87 0%, #60EFFF 100%); width: {portability_score}%; height: 100%; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 255, 135, 0.3);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        predict_clicked = st.button("Generate Value Estimation", key="predict_action")

    # Handle prediction calculation
    if predict_clicked:
        touchscreen_val = 1 if touchscreen == "Yes" else 0
        ips_val = 1 if ips == "Yes" else 0
        X_res = int(resolution.split("x")[0])
        Y_res = int(resolution.split("x")[1])
        ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size
        
        query = pd.DataFrame({
            "Company": [company],
            "TypeName": [type_name],
            "Ram": [ram],
            "Weight": [weight],
            "TouchScreen": [touchscreen_val],
            "Ips": [ips_val],
            "ppi": [ppi],
            "Cpu brand": [cpu],
            "HDD": [hdd],
            "SSD": [ssd],
            "Gpu brand": [gpu],
            "os": [os_val]
        })
        
        try:
            prediction = np.exp(pipe.predict(query)[0])
            
            # Formulate suggested profile
            if type_name == 'Gaming' or (gpu == 'Nvidia' and ram >= 16):
                profile = "Extreme Gaming / Creator Beast"
                desc = "Excellent graphic capabilities and premium multitasking. Built for demanding software, creative pipelines, and high-FPS gaming."
                accent_color = "#9B51E0"
                appraisal = "Enthusiast Spec"
            elif type_name == 'Workstation' or (ram >= 16 and (cpu == 'Intel Core i7' or cpu == 'AMD Processor')):
                profile = "Professional Powerhouse Workstation"
                desc = "Designed for high-intensity compilation, virtualization, data processing, and enterprise workloads."
                accent_color = "#4FACFE"
                appraisal = "Professional Spec"
            elif type_name == 'Ultrabook' or (type_name == '2 in 1 Convertible' and weight <= 1.4):
                profile = "Sleek Ultra-Premium Portable"
                desc = "Focuses on portability, battery endurance, and premium build. Perfect for frequent travelers and corporate presentations."
                accent_color = "#00F2FE"
                appraisal = "Premium Portable"
            elif ram >= 8 and (cpu == 'Intel Core i5' or cpu == 'Intel Core i7'):
                profile = "Balanced Modern Office / Dev Machine"
                desc = "Dynamic specifications offering smooth coding, browser multitasking, and comfortable productivity workflows."
                accent_color = "#00FF87"
                appraisal = "Mainstream Spec"
            else:
                profile = "Standard Entry-Level Companion"
                desc = "Best suited for standard tasks including document processing, web navigation, research, and casual media streaming."
                accent_color = "#F59E0B"
                appraisal = "Value Spec"
                
            storage_desc = ""
            if ssd > 0 and hdd > 0:
                storage_desc = f"{ssd}GB SSD + {hdd}GB HDD"
            elif ssd > 0:
                storage_desc = f"{ssd}GB SSD"
            else:
                storage_desc = f"{hdd}GB HDD"
                
            # Percentile calculation
            price_val = int(prediction)
            percentile = (df['Price'] < price_val).mean() * 100
                
            # Update state
            st.session_state.prediction_data = {
                "price": price_val,
                "low_range": int(price_val * 0.94),
                "high_range": int(price_val * 1.06),
                "profile": profile,
                "desc": desc,
                "company": company,
                "cpu": cpu,
                "gpu": gpu,
                "ram": ram,
                "storage_desc": storage_desc,
                "accent_color": accent_color,
                "appraisal": appraisal,
                "percentile": percentile
            }
        except Exception as e:
            st.error(f"Error predicting price: {e}")

    # Display prediction card and Plotly charts if prediction exists in state
    if st.session_state.prediction_data is not None:
        pred = st.session_state.prediction_data
        
        st.html(f"""
        <div style="
            background: linear-gradient(135deg, rgba(15, 19, 38, 0.95) 0%, rgba(6, 8, 14, 0.98) 100%);
            border: 2px solid {pred['accent_color']};
            border-radius: 16px;
            padding: 24px;
            margin-top: 15px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6), 0 0 25px {pred['accent_color']}30;
            text-align: center;
        ">
            <span style="font-family: 'Orbitron', sans-serif; color: #94A3B8; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; display: block; margin-bottom: 5px;">Estimated Market Value</span>
            <span style="font-family: 'Orbitron', sans-serif; color: {pred['accent_color']}; font-size: 38px; font-weight: 900; text-shadow: 0 0 20px {pred['accent_color']}50; display: block; margin-bottom: 5px;">₹ {pred['price']:,}</span>
            <span style="color: #64748B; font-size: 13px; display: block; margin-bottom: 12px; font-weight: 500;">
                Appraisal Range: <b>₹ {pred['low_range']:,} - ₹ {pred['high_range']:,}</b>
            </span>
            
            <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 14px; margin-bottom: 15px; text-align: left;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-family: 'Orbitron', sans-serif; color: #FFFFFF; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;">🎯 {pred['profile']}</span>
                    <span style="background: {pred['accent_color']}15; border: 1px solid {pred['accent_color']}40; color: {pred['accent_color']}; font-size: 10px; padding: 2px 8px; border-radius: 20px; font-weight: 700; font-family: 'Orbitron', sans-serif;">{pred['appraisal']}</span>
                </div>
                <p style="font-size: 12px; color: #94A3B8; margin: 0; line-height: 1.5;">{pred['desc']}</p>
                <div style="margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.04); padding-top: 8px; display: flex; justify-content: space-between; font-size: 11px; color: #64748B;">
                    <span>Market Position:</span>
                    <span style="color: #E2E8F0; font-weight: bold;">Valued higher than {pred['percentile']:.1f}% of configurations</span>
                </div>
            </div>
            
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 6px;">
                <span style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); color: #E2E8F0; font-size: 10px; padding: 4px 10px; border-radius: 12px; font-weight: 600;">{pred['company']}</span>
                <span style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); color: #E2E8F0; font-size: 10px; padding: 4px 10px; border-radius: 12px; font-weight: 600;">{pred['cpu']}</span>
                <span style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); color: #E2E8F0; font-size: 10px; padding: 4px 10px; border-radius: 12px; font-weight: 600;">{pred['gpu']}</span>
                <span style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); color: #E2E8F0; font-size: 10px; padding: 4px 10px; border-radius: 12px; font-weight: 600;">{pred['ram']}GB RAM</span>
                <span style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); color: #E2E8F0; font-size: 10px; padding: 4px 10px; border-radius: 12px; font-weight: 600;">{pred['storage_desc']}</span>
            </div>
        </div>
        """)
            
    else:
        # Display Awaiting State Card
        st.markdown(f"""
        <div style="
            background: rgba(15, 19, 38, 0.25);
            border: 1px dashed rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 40px 24px;
            margin-top: 15px;
            text-align: center;
        ">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(0, 242, 254, 0.3)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 16px; animation: pulse 2s infinite ease-in-out;">
                <rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect>
                <rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect>
                <line x1="6" y1="6" x2="6.01" y2="6"></line>
                <line x1="6" y1="18" x2="6.01" y2="18"></line>
                <line x1="10" y1="6" x2="10.01" y2="6"></line>
                <line x1="10" y1="18" x2="10.01" y2="18"></line>
            </svg>
            <h5 style="color: #E2E8F0; margin-bottom: 8px; font-family: 'Orbitron', sans-serif;">Awaiting Valuation</h5>
            <p style="color: #64748B; font-size: 12px; margin: 0; line-height: 1.5;">
                Configure the specifications in the selector panel and click "Generate Value Estimation" to run the forecasting engine.
            </p>
        </div>
        """, unsafe_allow_html=True)
