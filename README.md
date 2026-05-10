# 🌩️ Cloudburst ML — AI-Powered Cloudburst Prediction System

Cloudburst ML is an advanced AI-powered weather intelligence platform designed to predict cloudburst events using atmospheric physics, machine learning, and real-time meteorological data.

The system combines:
- ⚡ Physics-based Machine Learning
- 🌍 Live NOAA GFS Forecast Data
- 📰 News & Social Media Verification
- 🤖 NLP-based Disaster Signal Detection
- 📊 Interactive Streamlit Dashboard
- 🚨 Fusion-based Alert Generation

Built to provide intelligent, real-time, and explainable cloudburst risk predictions. :contentReference[oaicite:0]{index=0}

---

# 🚀 Features

| Feature | Description |
|---|---|
| 🌦️ Atmospheric Physics Model | XGBoost model trained on meteorological parameters |
| 🌍 Live GFS Forecast Integration | Uses NOAA GFS weather forecasts in real-time |
| 🧠 Machine Learning Pipeline | Detects cloudburst-prone atmospheric conditions |
| 📰 Verification Layer | Scrapes news & Reddit for disaster verification |
| ⚡ Fusion Alert System | Combines science predictions + NLP verification |
| 📍 Geolocation Support | Reverse geocoding for location identification |
| 📊 Interactive Dashboard | Streamlit-based real-time monitoring UI |
| 🚨 Intelligent Alerts | Risk-level based cloudburst warnings |

---

# 🧠 System Architecture

```text
Historical Weather Data
            │
            ▼
Training Dataset Generation
            │
            ▼
XGBoost Model Training
            │
            ▼
Real-Time NOAA GFS Forecast Data
            │
            ▼
Cloudburst Probability Prediction
            │
            ▼
News & Social Media Verification
            │
            ▼
Fusion Decision Engine
            │
            ▼
Final Alert Generation
```

---

# ⚙️ Tech Stack

## 🔹 Machine Learning
- Python
- XGBoost
- Scikit-learn
- SMOTE
- Pandas
- NumPy

## 🔹 Weather Data
- NOAA GFS Forecast Data
- Herbie
- Xarray

## 🔹 NLP & Verification
- Transformers
- DistilBERT
- BeautifulSoup
- Feedparser

## 🔹 Visualization & Dashboard
- Streamlit
- Plotly
- Folium
- Streamlit-Folium

## 🔹 Other Tools
- Reverse Geocoder
- Requests
- Git & GitHub

---

# 📂 Project Structure

```text
Cloudburst_ML/
│
├── app.py
├── requirements.txt
├── packages.txt
│
├── 00_create_target_file.py
├── 01_train_model.py
├── 02_predict_my_location.py
├── 02_run_prediction.py
├── 03_verify_my_location.py
├── 03_run_verification.py
├── 04_alert_my_location.py
├── 04_run_fusion_alert.py
│
├── verification_scraper_module.py
├── model_columns.json
│
├── science_predictions.csv
├── verification_results.json
├── my_location_info.json
│
├── xgb_cloudburst_model.json
├── xgb_flood_model.json
│
└── herbie_cache/
```

---

# 🔬 Machine Learning Model

## 📌 Training Features

The model uses atmospheric physics-based features:

```json
[
  "lat",
  "lon",
  "dew2m",
  "latent_flux",
  "sensible_flux",
  "cloud_cover",
  "cloud_liquid",
  "wind_speed"
]
```

---

## 📌 Model Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost |
| Event Threshold | 50mm Rainfall |
| Imbalance Handling | SMOTE |
| Max Trees | 2000 |
| Early Stopping | 10 rounds |

---

# 🌍 Real-Time Prediction Pipeline

## Step 1 — Training Dataset Generation
Creates labeled training data using historical rainfall and weather features.

```bash
python 00_create_target_file.py
```

---

## Step 2 — Train Models

```bash
python 01_train_model.py
```

Generated Models:
- `xgb_cloudburst_model.json`
- `xgb_flood_model.json`

---

## Step 3 — Predict Cloudburst Risk

```bash
python 02_predict_my_location.py
```

Uses:
- Real-time NOAA GFS forecast data
- Physics-based atmospheric features
- Location-based forecasting

---

## Step 4 — Verification Layer

```bash
python 03_verify_my_location.py
```

The system:
- Scrapes news articles
- Searches Reddit discussions
- Uses NLP classification for disaster verification

---

## Step 5 — Fusion Alert Engine

```bash
python 04_alert_my_location.py
```

Final decision formula:

```text
Final Score =
(Science Prediction × 70%)
+
(News Verification × 30%)
```

---

# 🚨 Alert Levels

| Score | Alert |
|---|---|
| ≥ 0.75 | 🚨 HIGH RISK |
| 0.50 - 0.75 | ⚠️ CAUTION |
| < 0.50 | ✅ LOW RISK |

---

# 📊 Interactive Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Dashboard Features:
- 📍 Real-time location predictions
- 🌩️ Cloudburst probability maps
- 📊 Interactive visualizations
- 📰 News verification results
- 🚨 Final alert decisions

---

# 📡 Live NOAA GFS Integration

The project uses:
- NOAA Global Forecast System
- 0.25° resolution forecast grids
- 6-hour update cycles
- Real-time atmospheric forecasting

Supported forecast cycles:
- 00z
- 06z
- 12z
- 18z

---

# 📁 Output Files

| File | Description |
|---|---|
| `science_predictions.csv` | Global prediction map |
| `verification_results.json` | NLP verification results |
| `my_location_info.json` | Prediction for selected location |
| `herbie_cache/` | Cached GFS weather data |

---

# ⚡ Quick Start

## 1️⃣ Clone Repository

```bash
git clone https://github.com/rohan-k-dev/cloudburst-prediction-model.git
cd cloudburst-prediction-model
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Train Models

```bash
python 01_train_model.py
```

---

## 4️⃣ Run Prediction

```bash
python 02_predict_my_location.py
```

---

## 5️⃣ Verify Predictions

```bash
python 03_verify_my_location.py
```

---

## 6️⃣ Generate Final Alerts

```bash
python 04_alert_my_location.py
```

---

## 7️⃣ Launch Dashboard

```bash
streamlit run app.py
```

---

# 🌟 Key Highlights

✅ Physics-based ML Prediction  
✅ Real-Time Weather Forecast Integration  
✅ Explainable AI Pipeline  
✅ NLP-Based Disaster Verification  
✅ Fusion-Based Alert Engine  
✅ Interactive Streamlit Dashboard  
✅ Global Location Prediction Support  
✅ Scalable & Modular Architecture  

---

# 🏆 Use Cases

- 🌩️ Extreme Rainfall Prediction
- 🌍 Disaster Monitoring
- 🚨 Early Warning Systems
- 🛰️ Weather Intelligence Platforms
- 🧠 AI + Climate Research
- 📊 Atmospheric Data Analysis

---

# 👨‍💻 Author

## Rohan Kumar

- 🎓 BMS College of Engineering (BMSCE)
- 💻 Full-Stack Developer | ML Engineer | Cloud Enthusiast

### 🌐 Connect With Me

- GitHub: https://github.com/rohan-k-dev
- LinkedIn: https://linkedin.com/in/rohan19725

---

# ⭐ Future Improvements

- Satellite Image Integration
- Deep Learning Weather Models
- Multi-Hazard Prediction
- Mobile Alert System
- Live Disaster Heatmaps
- Real-Time Notification Service

---

# 📜 License

This project is developed for research, educational, and innovation purposes.
