import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime, timedelta
from herbie import Herbie
import warnings
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- IMPORT VERIFICATION MODULE (Preserved) ---
try:
    import verification_scraper_module as scraper_module
except ImportError:
    # We allow the app to run even if this is missing, just for UI demo purposes
    pass

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="N.D.I.S | Cloudburst Intel",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed for a full dashboard feel
)

# --- ADVANCED CSS STYLING (THEME ENGINE) ---
st.markdown("""
    <style>
    /* GLOBAL THEME SETTINGS */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Roboto+Mono:wght@400;700&display=swap');
    
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(at 50% 0%, #1a1a2e 0%, #000000 70%);
        color: #e0e0e0;
        font-family: 'Rajdhani', sans-serif;
    }

    /* REMOVE STREAMLIT CHROME */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* HEADERS */
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }

    /* GLASSMORPHISM CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }

    /* METRIC VALUE STYLES */
    .metric-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00f2fe;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* INPUT FIELDS */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 5px;
        font-family: 'Roboto Mono', monospace;
    }
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.3);
    }

    /* BUTTONS */
    .stButton>button {
        background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border: 1px solid #00f2fe;
        color: #00f2fe;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: bold;
        width: 100%;
        padding: 12px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #00f2fe;
        color: #000;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.6);
        transform: translateY(-2px);
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #666;
        font-family: 'Rajdhani', sans-serif;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #00f2fe;
        border-bottom: 2px solid #00f2fe;
    }

    /* CUSTOM ALERTS */
    .status-safe {
        border-left: 5px solid #00f2fe;
        background: linear-gradient(90deg, rgba(0,242,254,0.1) 0%, transparent 100%);
        padding: 15px;
    }
    .status-caution {
        border-left: 5px solid #ffd89b;
        background: linear-gradient(90deg, rgba(255,216,155,0.1) 0%, transparent 100%);
        padding: 15px;
    }
    .status-danger {
        border-left: 5px solid #ff416c;
        background: linear-gradient(90deg, rgba(255,65,108,0.2) 0%, transparent 100%);
        animation: pulse-red 2s infinite;
        padding: 15px;
    }

    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 65, 108, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 65, 108, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 65, 108, 0); }
    }
    
    /* CUSTOM SCROLLBAR */
    ::-webkit-scrollbar {
        width: 10px;
        background: #000;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- BACKEND LOGIC (PRESERVED VERBATIM) ---

@st.cache_resource
def load_xgboost_model():
    """Load the XGBoost model"""
    try:
        if os.path.exists('xgb_flood_model.json'):
            bst = xgb.Booster()
            bst.load_model('xgb_flood_model.json')
            return bst
        else:
            st.error("❌ xgb_flood_model.json not found!")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_model_columns():
    try:
        if os.path.exists('model_columns.json'):
            with open('model_columns.json', 'r') as f:
                return json.load(f)
        else:
            st.error("❌ model_columns.json not found!")
            return None
    except Exception as e:
        st.error(f"Error loading model columns: {e}")
        return None

def get_safe_gfs_date():
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    if hour >= 22: run_date = now_utc.replace(hour=18, minute=0, second=0, microsecond=0)
    elif hour >= 16: run_date = now_utc.replace(hour=12, minute=0, second=0, microsecond=0)
    elif hour >= 10: run_date = now_utc.replace(hour=6, minute=0, second=0, microsecond=0)
    elif hour >= 4: run_date = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    else: run_date = (now_utc - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
    return run_date

def fetch_gfs_data_for_location(lat, lon, radius=0.5):
    try:
        run_date = get_safe_gfs_date()
        def try_fetch(target_date):
            H = Herbie(date=target_date, model='gfs', product='pgrb2.0p25', fxx=6, save_dir='herbie_cache', verbose=True)
            min_lat, max_lat = lat - radius, lat + radius
            min_lon, max_lon = lon - radius, lon + radius
            variables = {
                'dpt': ':DPT:2 m above', 'lhtfl': ':LHTFL:surface', 'shtfl': ':SHTFL:surface',
                'tcdc': ':TCDC:entire atmosphere', 'cwat': ':CWAT:', 'ugrd': ':UGRD:10 m above',
                'vgrd': ':VGRD:10 m above', 'apcp': ':APCP:surface', 'tmp': ':TMP:2 m above'
            }
            data_dict = {}
            coords_ref = None
            for var, search in variables.items():
                try:
                    ds = H.xarray(search, remove_grib=False)
                    if isinstance(ds, list): ds = ds[0]
                    ds = ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
                    val_key = list(ds.data_vars.keys())[0]
                    data_dict[var] = ds[val_key].values.flatten()
                    if coords_ref is None:
                        lats = ds.latitude.values
                        lons = ds.longitude.values
                        lon_grid, lat_grid = np.meshgrid(lons, lats)
                        coords_ref = {'lat': lat_grid.flatten(), 'lon': lon_grid.flatten()}
                except:
                    if var in ['apcp', 'tmp']: data_dict[var] = None
                    pass
            return data_dict, coords_ref

        all_data, coords = try_fetch(run_date)
        if not all_data or coords is None:
            fallback_date = run_date - timedelta(hours=6)
            all_data, coords = try_fetch(fallback_date)
        
        if not all_data or coords is None: return None
        
        df = pd.DataFrame(coords)
        for v, d in all_data.items(): 
            if d is not None: df[v] = d
        if 'apcp' not in df.columns: df['apcp'] = 0.0
        if 'tmp' not in df.columns: df['tmp'] = 290.0
        return df
    except Exception as e:
        return None

def fetch_india_wide_data():
    try:
        run_date = get_safe_gfs_date()
        def try_fetch_india(target_date):
            H = Herbie(date=target_date, model='gfs', product='pgrb2.0p25', fxx=6, save_dir='herbie_cache')
            variables = {
                'dpt': ':DPT:2 m above', 'lhtfl': ':LHTFL:surface', 'shtfl': ':SHTFL:surface',
                'tcdc': ':TCDC:entire atmosphere', 'cwat': ':CWAT:', 'ugrd': ':UGRD:10 m above',
                'vgrd': ':VGRD:10 m above', 'apcp': ':APCP:surface', 'tmp': ':TMP:2 m above'
            }
            data_dict = {}
            coords_ref = None
            for v, s in variables.items():
                try:
                    ds = H.xarray(s, remove_grib=False)
                    if isinstance(ds, list): ds = ds[0]
                    ds = ds.sel(latitude=slice(37, 6), longitude=slice(68, 98))
                    val_key = list(ds.data_vars.keys())[0]
                    data_dict[v] = ds[val_key].values.flatten()
                    if coords_ref is None:
                        lats = ds.latitude.values
                        lons = ds.longitude.values
                        lon_grid, lat_grid = np.meshgrid(lons, lats)
                        coords_ref = {'lat': lat_grid.flatten(), 'lon': lon_grid.flatten()}
                except:
                    if v in ['apcp', 'tmp']: data_dict[v] = None
                    pass
            return data_dict, coords_ref

        all_data, coords = try_fetch_india(run_date)
        if not all_data or coords is None:
            fallback_date = run_date - timedelta(hours=6)
            all_data, coords = try_fetch_india(fallback_date)

        if coords is None: return None
        df = pd.DataFrame(coords)
        for v, d in all_data.items():
            if d is not None: df[v] = d
        if 'apcp' not in df.columns: df['apcp'] = 0.0
        if 'tmp' not in df.columns: df['tmp'] = 290.0
        return df
    except Exception as e:
        st.error(f"Error fetching India-wide data: {e}")
        return None

def process_data_for_prediction(df_raw):
    df = df_raw.copy()
    rename_map = {'dpt': 'dew2m', 'lhtfl': 'latent_flux', 'shtfl': 'sensible_flux', 'tcdc': 'cloud_cover', 'cwat': 'cloud_liquid'}
    df = df.rename(columns=rename_map)
    if 'latent_flux' in df.columns: df['latent_flux'] *= -3600
    if 'sensible_flux' in df.columns: df['sensible_flux'] *= -3600
    if 'cloud_cover' in df.columns: df['cloud_cover'] /= 100.0
    if 'ugrd' in df.columns and 'vgrd' in df.columns: df['wind_speed'] = np.sqrt(df['ugrd']**2 + df['vgrd']**2)
    return df

def predict_cloudburst_location(lat, lon, location_name):
    model = load_xgboost_model()
    model_cols = load_model_columns()
    if model is None or model_cols is None: return None
    
    with st.spinner('🛰️ ESTABLISHING SATELLITE UPLINK...'):
        df_raw = fetch_gfs_data_for_location(lat, lon)
    
    if df_raw is None:
        st.error("Failed to fetch GFS data (Network or Source Error)")
        return None
    
    df = process_data_for_prediction(df_raw)
    for c in model_cols:
        if c not in df.columns: df[c] = 0.0
    
    dmatrix = xgb.DMatrix(df[model_cols])
    probs = model.predict(dmatrix)
    df['probability'] = probs
    
    best_idx = df['probability'].idxmax()
    max_prob = df['probability'].max()
    local_rain = df.iloc[best_idx].get('apcp', 0.0)
    local_temp = df.iloc[best_idx].get('tmp', 290.0)
    
    final_prob = max_prob
    veto_applied = False
    veto_reason = ""
    
    if local_rain < 0.5 and max_prob > 0.5:
        final_prob = 0.0
        veto_applied = True
        veto_reason = "No precipitation detected by satellite"
    if local_temp < 283.15 and max_prob > 0.5:
        final_prob = 0.0
        veto_applied = True
        veto_reason = "Temperature too cold (likely fog)"
    
    result = {
        'location_name': location_name, 'lat': lat, 'lon': lon,
        'raw_probability': float(max_prob), 'final_probability': float(final_prob),
        'local_rain': float(local_rain), 'local_temp': float(local_temp),
        'veto_applied': veto_applied, 'veto_reason': veto_reason,
        'timestamp': datetime.now().isoformat()
    }
    return result

def run_verification(location_name):
    try:
        # Check if scraper module exists
        if 'scraper_module' not in globals():
             return 0.0, 0, []

        scraper_module.clear_cache_for_new_location()
        scraper = scraper_module.VerificationScraper(
            location=location_name, hours_lookback=48, enable_fallback=True, min_articles_threshold=5
        )
        data = scraper.scrape_all()
        if data.empty: return 0.0, 0, []
        
        danger_keywords = ['flood', 'cloudburst', 'heavy rain', 'downpour', 'waterlogging', 'evacuation', 'alert', 'warning']
        data['text_blob'] = data['title'].fillna('') + " " + data['summary'].fillna('')
        scores = []
        top_articles = []
        
        for idx, row in data.iterrows():
            text = row['text_blob'].lower()
            keyword_count = sum(1 for kw in danger_keywords if kw in text)
            if keyword_count >= 2:
                score = min(keyword_count / len(danger_keywords), 1.0)
                scores.append(score)
                top_articles.append({'title': row['title'], 'source': row['source'], 'score': score})
        
        text_score = np.mean(scores) if scores else 0.0
        return float(text_score), len(data), sorted(top_articles, key=lambda x: x['score'], reverse=True)[:5]
    except Exception as e:
        return 0.0, 0, []

# --- UI COMPONENT FUNCTIONS ---

def render_hero_header():
    st.markdown("""
        <div style="text-align: center; margin-bottom: 40px; padding: 20px; border-bottom: 1px solid rgba(255,255,255,0.1);">
            <h1 style="font-size: 3.5rem; margin-bottom: 0;">N.D.I.S</h1>
            <p style="color: #00f2fe; font-size: 1.2rem; letter-spacing: 3px; font-family: 'Roboto Mono', monospace;">NATIONAL DISASTER INTELLIGENCE SYSTEM</p>
        </div>
    """, unsafe_allow_html=True)

def render_metric_card(label, value, subtext="", color="#00f2fe"):
    st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color: {color};">{value}</div>
            <div style="font-size: 0.8rem; color: #666; margin-top: 5px;">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

def render_status_card(score, threshold=0.75):
    if score >= threshold:
        css_class = "status-danger"
        title = "CRITICAL ALERT: CLOUDBURST IMMINENT"
        desc = "High probability of extreme weather event. Immediate mobilization advised."
        icon = "🚨"
    elif score >= 0.5:
        css_class = "status-caution"
        title = "WARNING: UNSTABLE ATMOSPHERE"
        desc = "Conditions are favorable for heavy rainfall. Monitor situation closely."
        icon = "⚠️"
    else:
        css_class = "status-safe"
        title = "STATUS NORMAL"
        desc = "No significant weather threats detected in the area."
        icon = "✅"
    
    st.markdown(f"""
        <div class="glass-card {css_class}">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="font-size: 3rem;">{icon}</div>
                <div>
                    <h3 style="margin: 0; font-size: 1.5rem;">{title}</h3>
                    <p style="margin: 0; color: #ddd;">{desc}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- MAIN APP STRUCTURE ---

def main():
    render_hero_header()
    
    tab1, tab2 = st.tabs(["LOCATION INTEL", "NATIONAL SCAN"])
    
    # === TAB 1: LOCATION INTELLIGENCE ===
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Input Section
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                location_name = st.text_input("TARGET SECTOR", value="Kavaratti", help="Enter City/Region Name")
            with col2:
                lat = st.number_input("LAT", value=9.5000, format="%.4f")
            with col3:
                lon = st.number_input("LON", value=73.0000, format="%.4f")
            with col4:
                st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True) # Spacer
                analyze = st.button("INITIATE SCAN")

        if analyze:
            st.markdown("---")
            
            # 1. Physics Analysis
            result = predict_cloudburst_location(lat, lon, location_name)
            
            if result:
                # Top Row Metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    render_metric_card("Precipitation", f"{result['local_rain']:.2f} mm", "Sat-Derived")
                with m2:
                    render_metric_card("Surface Temp", f"{result['local_temp']-273.15:.1f}°C", "GFS Model")
                with m3:
                    render_metric_card("AI Model Confidence", f"{result['raw_probability']:.1%}", "XGBoost")
                with m4:
                    render_metric_card("Veto Status", "ACTIVE" if result['veto_applied'] else "INACTIVE", 
                                     result['veto_reason'] if result['veto_applied'] else "No Restrictions",
                                     color="#ff416c" if result['veto_applied'] else "#00f2fe")

                science_prob = result['final_probability']

                # 2. News Verification
                with st.spinner('📡 INTERCEPTING NEWS FEEDS...'):
                    text_score, article_count, top_articles = run_verification(location_name)
                
                # Fusion Calculation
                SCIENCE_WEIGHT, TEXT_WEIGHT, THRESHOLD = 0.70, 0.30, 0.75
                final_score = (science_prob * SCIENCE_WEIGHT) + (text_score * TEXT_WEIGHT)
                
                # 3. Main Dashboard Display
                c_left, c_right = st.columns([2, 1])
                
                with c_left:
                    st.markdown("### SITUATION REPORT")
                    render_status_card(final_score, THRESHOLD)
                    
                    if top_articles:
                        st.markdown("#### INTELLIGENCE FEEDS")
                        for article in top_articles:
                            score_color = "#ff416c" if article['score'] > 0.6 else "#ffd89b"
                            st.markdown(f"""
                                <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid {score_color}">
                                    <span style="color: {score_color}; font-weight: bold;">[{article['score']:.0%}]</span> 
                                    {article['title']} <span style="opacity: 0.5; font-size: 0.8rem">({article['source']})</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No active chatter detected regarding flooding in this sector.")

                with c_right:
                    st.markdown("### RISK GAUGE")
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = final_score * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#ff416c" if final_score > 0.75 else ("#ffd89b" if final_score > 0.5 else "#00f2fe")},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "#333",
                            'steps': [
                                {'range': [0, 50], 'color': "rgba(0, 242, 254, 0.1)"},
                                {'range': [50, 75], 'color': "rgba(255, 216, 155, 0.1)"},
                                {'range': [75, 100], 'color': "rgba(255, 65, 108, 0.1)"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': THRESHOLD * 100
                            }
                        }
                    ))
                    fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Rajdhani"})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    render_metric_card("FUSION SCORE", f"{final_score:.1%}", "Combined Science + News")

    # === TAB 2: NATIONAL SCAN ===
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        col_scan, col_info = st.columns([1, 3])
        with col_scan:
            scan_btn = st.button("START NATIONAL SCAN")
        with col_info:
            st.markdown("<p style='padding-top: 10px; color: #666;'>Scans high-resolution GFS grids across the Indian Subcontinent.</p>", unsafe_allow_html=True)

        if scan_btn:
            model = load_xgboost_model()
            model_cols = load_model_columns()
            
            if model and model_cols:
                with st.spinner('🛰️ DOWNLOADING INDIA-WIDE DATA PACKETS...'):
                    df_raw = fetch_india_wide_data()
                
                if df_raw is not None:
                    with st.spinner('⚙️ PROCESSING NEURAL NETWORKS...'):
                        df = process_data_for_prediction(df_raw)
                        for c in model_cols:
                            if c not in df.columns: df[c] = 0.0
                        
                        dmatrix = xgb.DMatrix(df[model_cols])
                        df['probability'] = model.predict(dmatrix)
                        
                        # Post-processing filters
                        df.loc[df['apcp'] < 0.2, 'probability'] = 0.0
                        df.loc[df['tmp'] < 283.15, 'probability'] = 0.0
                        
                        top_10 = df.nlargest(10, 'probability')[['lat', 'lon', 'probability', 'apcp', 'tmp']]
                        
                        if top_10.iloc[0]['probability'] > 0.0:
                            st.markdown("### 🗺️ IDENTIFIED HOTSPOTS")
                            
                            # Dark Map Styling
                            fig = px.scatter_mapbox(
                                top_10, lat='lat', lon='lon', size='probability', color='probability',
                                color_continuous_scale=['#00f2fe', '#ffd89b', '#ff416c'],
                                size_max=25, zoom=4, height=600,
                                hover_data={'probability': ':.2%', 'apcp': ':.2f', 'tmp': ':.1f'}
                            )
                            fig.update_layout(
                                mapbox_style="carto-darkmatter",
                                margin={"r":0,"t":0,"l":0,"b":0},
                                paper_bgcolor="rgba(0,0,0,0)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("### 📊 TELEMETRY DATA")
                            
                            # Styled Dataframe
                            display_df = top_10.copy()
                            display_df.columns = ['LAT', 'LON', 'RISK', 'RAIN (mm)', 'TEMP (K)']
                            st.dataframe(display_df, use_container_width=True)
                        else:
                            st.markdown("""
                                <div class="glass-card status-safe">
                                    <h3>SCAN COMPLETE</h3>
                                    <p>No anomalous cloudburst signatures detected across the subcontinent.</p>
                                </div>
                            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
