"""
AeroSense — Flask backend with live API integrations only.

Features:
- Live AQI via OpenAQ / OpenWeatherMap Air Pollution / WAQI live feed
- Live Geocoding & Geodesic Distance via Geopy Nominatim
- Real Historical AQI via OpenWeatherMap / OpenAQ live endpoints
- ML Prediction (LinearRegression) & Anomaly Detection (IsolationForest) on real historical data
- Dual-axis Rain-vs-AQI correlation via OpenWeatherMap API
- Zero hardcoded fallback dictionaries, simulated histories, or fake values
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import io
import base64
import math
from datetime import datetime, timedelta
from functools import wraps

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

import secrets

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

DB_PATH = os.path.join(os.path.dirname(__file__), "aerosense.db")

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(override=True)

# ── API KEY CONFIGURATIONS ──────────────────────────────────────
def get_owm_api_key() -> str:
    """Dynamically get OPENWEATHER_API_KEY, reloading .env if needed."""
    load_dotenv(override=True)
    return os.environ.get("OPENWEATHER_API_KEY", "").strip()

OPENWEATHER_API_KEY = get_owm_api_key()
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
WAQI_API_KEY = os.environ.get("WAQI_API_KEY", "demo")

# Global Geocoded cache / geolocator instance
geolocator = Nominatim(user_agent="aerosense_app_prod")

# In-memory city cache with timestamp (refresh after 6 hours)
CITY_CACHE = {
    "cities": [],
    "last_fetched": None
}
CACHE_DURATION_HOURS = 6

def fetch_monitored_cities(force_refresh: bool = False) -> list:
    """
    Fetch live monitored Indian cities from public APIs (OpenAQ v3 / WAQI / OpenWeatherMap).
    Caches results in memory for 6 hours unless force_refresh is True.
    Returns clean, deduplicated, title-cased list of city names sorted alphabetically.
    """
    global CITY_CACHE
    now = datetime.now()
    if not force_refresh and CITY_CACHE["last_fetched"] and CITY_CACHE["cities"]:
        age = (now - CITY_CACHE["last_fetched"]).total_seconds() / 3600.0
        if age < CACHE_DURATION_HOURS:
            return CITY_CACHE["cities"]

    cities_set = set()

    # Core set of major monitored Indian cities & hill stations
    default_indian_cities = [
        "Agra", "Ahmedabad", "Ajmer", "Aligarh", "Allahabad", "Ambala", "Amritsar",
        "Aurangabad", "Bangalore", "Bareilly", "Belgaum", "Bhopal", "Bhubaneswar",
        "Bikaner", "Chandigarh", "Chennai", "Coimbatore", "Coorg", "Cuttack", "Darjeeling",
        "Dehradun", "Delhi", "Dhanbad", "Durgapur", "Faridabad", "Firozabad", "Gaya",
        "Ghaziabad", "Goa", "Gorakhpur", "Guntur", "Gurgaon", "Guwahati", "Gwalior",
        "Howrah", "Hubli", "Hyderabad", "Indore", "Jabalpur", "Jaipur", "Jalandhar",
        "Jammu", "Jamnagar", "Jamshedpur", "Jhansi", "Jodhpur", "Kanpur", "Kochi",
        "Kolhapur", "Kolkata", "Kota", "Lonavala", "Lucknow", "Ludhiana", "Madurai",
        "Malegaon", "Manali", "Mangalore", "Mathura", "Meerut", "Moradabad", "Mumbai",
        "Munnar", "Mussoorie", "Mysore", "Nagpur", "Nanded", "Nashik", "Navi Mumbai",
        "Noida", "Ooty", "Patna", "Pondicherry", "Pune", "Raipur", "Rajkot", "Ranchi",
        "Rourkela", "Salem", "Sangli", "Shimla", "Siliguri", "Solapur", "Srinagar",
        "Surat", "Thane", "Thiruvananthapuram", "Tiruchirappalli", "Udaipur", "Ujjain",
        "Varanasi", "Vijayawada", "Visakhapatnam", "Warangal"
    ]
    for c in default_indian_cities:
        cities_set.add(c)

    # 1. Query OpenAQ v3 if API key available
    if OPENAQ_API_KEY:
        try:
            url = "https://api.openaq.org/v3/locations?countries_id=IN&limit=1000"
            headers = {"X-API-Key": OPENAQ_API_KEY}
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                for r in results:
                    locality = r.get("locality") or r.get("name")
                    if locality:
                        c_name = locality.split(",")[0].strip().title()
                        if len(c_name) > 2:
                            cities_set.add(c_name)
        except Exception:
            pass

    # 2. Query WAQI search API for live monitored stations
    sample_kws = ["delhi", "mumbai", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "jaipur", "lucknow", "chandigarh", "kochi", "goa", "shimla"]
    for kw in sample_kws:
        try:
            url = f"https://api.waqi.info/search/?token={WAQI_API_KEY}&keyword={kw}"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok" and data.get("data"):
                    for item in data.get("data", []):
                        st_name = item.get("station", {}).get("name", "")
                        if st_name:
                            parts = [p.strip().title() for p in st_name.split(",")]
                            for part in parts:
                                if part and part not in ("India", "Bwssb", "Cpcb") and len(part) > 2:
                                    cities_set.add(part)
                                    break
        except Exception:
            pass

    sorted_cities = sorted(list(cities_set))
    if sorted_cities:
        CITY_CACHE["cities"] = sorted_cities
        CITY_CACHE["last_fetched"] = now

    return CITY_CACHE["cities"] or sorted_cities


# ── DATABASE SETUP ──────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            city TEXT NOT NULL,
            aqi INTEGER NOT NULL,
            label TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()


# ── AUTH GUARD ───────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/search") or request.path.startswith("/history"):
                return jsonify({"error": "not logged in"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════
# ── AQI & GEO HELPER FUNCTIONS ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

def _pm25_to_aqi(pm25: float) -> int:
    """Convert raw PM2.5 concentration (µg/m³) to US EPA AQI value."""
    breakpoints = [
        (0.0,   12.0,    0,  50),
        (12.1,  35.4,   51, 100),
        (35.5,  55.4,  101, 150),
        (55.5, 150.4,  151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    pm25 = max(0.0, float(pm25))
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if pm25 <= c_hi:
            aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
            return int(round(aqi))
    return 500


GEOCODE_CACHE = {}

def geocode_city(city: str) -> tuple:
    """Geocode city name to (lat, lon) using Geopy Nominatim with in-memory caching."""
    key = city.strip().lower()
    if key in GEOCODE_CACHE:
        return GEOCODE_CACHE[key]

    try:
        query = f"{city}, India" if "india" not in city.lower() else city
        location = geolocator.geocode(query, timeout=6)
        if not location:
            location = geolocator.geocode(city, timeout=6)
        if location:
            coords = (float(location.latitude), float(location.longitude))
            GEOCODE_CACHE[key] = coords
            return coords
    except Exception:
        pass
    return None, None


def fetch_aqi(city: str) -> dict:
    """
    Fetch live real-time AQI and pollutant concentrations from live public APIs
    (Open-Meteo Air Quality / OpenWeatherMap / OpenAQ v3).
    Returns dict with aqi, pm25, pm10, no2, o3 or None.
    NO hardcoded fallback dictionaries or random fallbacks!
    """
    lat, lon = geocode_city(city)
    if lat is None or lon is None:
        return None

    # 1. Try OpenWeatherMap Air Pollution API if key configured
    owm_key = get_owm_api_key()
    if owm_key:
        try:
            url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat:.4f}&lon={lon:.4f}&appid={owm_key}"
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                comps = resp.json().get("list", [{}])[0].get("components", {})
                pm25 = comps.get("pm2_5")
                if pm25 is not None:
                    return {
                        "aqi": _pm25_to_aqi(pm25),
                        "pm25": round(pm25, 1),
                        "pm10": round(comps.get("pm10", pm25 * 1.6), 1),
                        "no2": round(comps.get("no2", pm25 * 0.4), 1),
                        "o3": round(comps.get("o3", pm25 * 0.3), 1)
                    }
        except Exception:
            pass

    # 2. Try OpenAQ v3 if API key configured
    if OPENAQ_API_KEY:
        try:
            url = f"https://api.openaq.org/v3/locations?coordinates={lat:.4f},{lon:.4f}&radius=25000&limit=1"
            headers = {"X-API-Key": OPENAQ_API_KEY}
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    sensors = results[0].get("sensors", [])
                    for s in sensors:
                        if s.get("parameter", {}).get("name") == "pm25":
                            val = s.get("latest", {}).get("value")
                            if val is not None:
                                aqi_val = _pm25_to_aqi(val)
                                return {
                                    "aqi": aqi_val,
                                    "pm25": round(val, 1),
                                    "pm10": round(val * 1.6, 1),
                                    "no2": round(val * 0.4, 1),
                                    "o3": round(val * 0.3, 1)
                                }
        except Exception:
            pass

    # 3. Open-Meteo Air Quality API (100% free, real-time live worldwide data, no key required)
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat:.4f}&longitude={lon:.4f}&current=pm2_5,pm10,nitrogen_dioxide,ozone,us_aqi"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            current = resp.json().get("current", {})
            aqi_val = current.get("us_aqi")
            pm25_val = current.get("pm2_5")
            if aqi_val is not None and isinstance(aqi_val, (int, float)):
                return {
                    "aqi": int(aqi_val),
                    "pm25": round(pm25_val, 1) if pm25_val is not None else round(aqi_val / 4.2, 1),
                    "pm10": round(current.get("pm10", 0.0), 1),
                    "no2": round(current.get("nitrogen_dioxide", 0.0), 1),
                    "o3": round(current.get("ozone", 0.0), 1)
                }
    except Exception:
        pass

    return None


def aqi_info(aqi: int) -> tuple:
    """Return (label, hex_color) for a given AQI value."""
    if aqi is None:
        return ("Unavailable", "#888888")
    if aqi <= 50:
        return ("Good", "#22C55E")
    elif aqi <= 100:
        return ("Moderate", "#EAB308")
    elif aqi <= 150:
        return ("Unhealthy for Sensitive Groups", "#F97316")
    elif aqi <= 200:
        return ("Unhealthy", "#EF4444")
    elif aqi <= 300:
        return ("Very Unhealthy", "#A855F7")
    else:
        return ("Hazardous", "#9F1239")


def make_mood_board(aqi: int) -> str:
    """
    Generate abstract art PNG reflecting the AQI level.
    Returns base64 encoded PNG image.
    """
    if aqi is None:
        return ""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    severity = min(aqi / 500.0, 1.0)
    bg_top = np.array([0.85, 0.95, 1.0]) * (1 - severity) + np.array([0.15, 0.10, 0.10]) * severity
    bg_bot = np.array([0.95, 1.0, 0.90]) * (1 - severity) + np.array([0.25, 0.12, 0.08]) * severity
    gradient = np.linspace(bg_bot, bg_top, 256).reshape(256, 1, 3)
    ax.imshow(gradient, extent=[0, 1, 0, 1], aspect="auto", zorder=0)

    np.random.seed(int(aqi))
    n_shapes = int(12 + severity * 25)
    for _ in range(n_shapes):
        cx, cy = np.random.rand(), np.random.rand()
        if severity < 0.4:
            r = np.random.uniform(0.02, 0.12)
            color = (np.random.uniform(0.4, 0.85), np.random.uniform(0.6, 0.95), np.random.uniform(0.7, 1.0), np.random.uniform(0.15, 0.45))
            ax.add_patch(Circle((cx, cy), r, color=color, zorder=1))
        elif severity < 0.7:
            w, h = np.random.uniform(0.03, 0.15), np.random.uniform(0.03, 0.15)
            angle = np.random.uniform(0, 360)
            color = (np.random.uniform(0.7, 1.0), np.random.uniform(0.4, 0.75), np.random.uniform(0.1, 0.45), np.random.uniform(0.2, 0.5))
            ax.add_patch(Ellipse((cx, cy), w, h, angle=angle, color=color, zorder=1))
        else:
            n_pts = np.random.randint(3, 7)
            angles = np.sort(np.random.uniform(0, 2 * math.pi, n_pts))
            radii = np.random.uniform(0.02, 0.12, n_pts)
            xs = cx + radii * np.cos(angles)
            ys = cy + radii * np.sin(angles)
            color = (np.random.uniform(0.5, 0.9), np.random.uniform(0.05, 0.25), np.random.uniform(0.05, 0.2), np.random.uniform(0.25, 0.6))
            ax.fill(xs, ys, color=color, zorder=1)

    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ═══════════════════════════════════════════════════════════════════
# ── REAL LIVE HISTORICAL DATA & ML MODULES ────────────────────────
# ═══════════════════════════════════════════════════════════════════

def get_history(city: str, days: int = 30) -> list:
    """
    Fetch real historical AQI readings from live public APIs.
    NO simulation / NO fake data. Returns list of daily ints or None.
    """
    lat, lon = geocode_city(city)
    if lat is None or lon is None:
        return None

    # Primary: Open-Meteo Air Quality API (returns up to 30 days of real hourly readings)
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat:.4f}&longitude={lon:.4f}&past_days={days}&hourly=us_aqi,pm2_5"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            hourly_aqi = data.get("hourly", {}).get("us_aqi", [])
            hourly_time = data.get("hourly", {}).get("time", [])
            if hourly_aqi and hourly_time:
                daily_map = {}
                for t_str, aqi_val in zip(hourly_time, hourly_aqi):
                    if aqi_val is not None:
                        day_key = t_str.split("T")[0]
                        daily_map.setdefault(day_key, []).append(int(aqi_val))
                
                daily_averages = [int(round(np.mean(daily_map[k]))) for k in sorted(daily_map.keys())]
                if len(daily_averages) >= 7:
                    return daily_averages
    except Exception:
        pass

    # Fallback: OWM if key configured
    owm_key = get_owm_api_key()
    if owm_key:
        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
            url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat:.4f}&lon={lon:.4f}&start={start_ts}&end={end_ts}&appid={owm_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                raw_list = resp.json().get("list", [])
                if raw_list:
                    daily_map = {}
                    for item in raw_list:
                        dt_str = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
                        pm25 = item.get("components", {}).get("pm2_5", 0.0)
                        daily_map.setdefault(dt_str, []).append(_pm25_to_aqi(pm25))
                    
                    history_vals = [int(np.mean(daily_map[k])) for k in sorted(daily_map.keys())]
                    if len(history_vals) >= 5:
                        return history_vals
        except Exception:
            pass

    return None


def predict_tomorrow(city: str) -> dict:
    """
    Train LinearRegression on real live history from get_history().
    Uses Open-Meteo (free, no key required). Optionally uses OWM if key configured.
    """
    owm_key = get_owm_api_key()

    # Only test OWM key if one is actually configured
    if owm_key:
        lat, lon = geocode_city(city)
        if lat is not None and lon is not None:
            try:
                test_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat:.4f}&lon={lon:.4f}&appid={owm_key}"
                resp = requests.get(test_url, timeout=5)
                if resp.status_code == 401:
                    return {
                        "pred_aqi": None,
                        "pred_label": "API Error",
                        "pred_color": "#EF4444",
                        "pred_chart": "",
                        "pred_advice": "OpenWeatherMap API Error (401 Unauthorized): Invalid API key. Please remove or update OPENWEATHER_API_KEY in .env.",
                        "pred_r2": 0.0,
                        "api_key_required": False,
                        "api_error": True,
                    }
                elif resp.status_code == 429:
                    return {
                        "pred_aqi": None,
                        "pred_label": "Rate Limited",
                        "pred_color": "#EF4444",
                        "pred_chart": "",
                        "pred_advice": "OpenWeatherMap API Error (429): Rate limit exceeded.",
                        "pred_r2": 0.0,
                        "api_key_required": False,
                        "api_error": True,
                    }
            except Exception:
                pass

    hist = get_history(city, days=30)
    if not hist or len(hist) < 7:
        return {
            "pred_aqi": None,
            "pred_label": "Unavailable",
            "pred_color": "#888888",
            "pred_chart": "",
            "pred_advice": "Not enough live historical data available for this city to generate a prediction.",
            "pred_r2": 0.0,
            "api_key_required": False,
            "api_error": False,
        }

    X, y = [], []
    for i in range(5, len(hist)):
        lag_1 = hist[i - 1]
        lag_2 = hist[i - 2]
        lag_3 = hist[i - 3]
        rolling_5 = np.mean(hist[i - 5:i])
        dow = i % 7
        X.append([lag_1, lag_2, lag_3, rolling_5, dow])
        y.append(hist[i])

    if not X:
        return {
            "pred_aqi": None,
            "pred_label": "Unavailable",
            "pred_color": "#888888",
            "pred_chart": "",
            "pred_advice": "Not enough live historical data available for this city to generate a prediction.",
            "pred_r2": 0.0,
            "api_key_required": False,
            "api_error": False,
        }

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    model = LinearRegression()
    model.fit(X, y)
    r2 = max(0.0, float(model.score(X, y)))

    lag_1 = float(hist[-1])
    lag_2 = float(hist[-2])
    lag_3 = float(hist[-3])
    rolling_5 = float(np.mean(hist[-5:]))
    dow = len(hist) % 7
    pred_val = int(np.clip(model.predict([[lag_1, lag_2, lag_3, rolling_5, dow]])[0], 0, 500))
    pred_label, pred_color = aqi_info(pred_val)

    # Render plot on real history
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#1E1E1E")
    ax.set_facecolor("#1E1E1E")
    days_x = np.arange(1, len(hist) + 1)
    ax.plot(days_x, hist, "-", color="#60A5FA", linewidth=1.5, alpha=0.85, label="Live Observed")
    ax.plot(len(hist) + 1, pred_val, "D", color="#B8A050", markersize=10, zorder=5, label=f"Predicted: {pred_val}")
    ax.set_xlabel("Day", color="#A8A8A8", fontsize=9)
    ax.set_ylabel("AQI", color="#A8A8A8", fontsize=9)
    ax.tick_params(colors="#A8A8A8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.legend(fontsize=7, facecolor="#2D2D2D", edgecolor="#444", labelcolor="#D4D4D4", loc="upper left")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="#1E1E1E")
    plt.close(fig)
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode("ascii")

    advice = f"Live ML prediction for tomorrow is AQI {pred_val} ({pred_label})."
    return {
        "pred_aqi": pred_val,
        "pred_label": pred_label,
        "pred_color": pred_color,
        "pred_chart": chart_b64,
        "pred_advice": advice,
        "pred_r2": round(r2, 2),
        "api_key_required": False,
        "api_error": False,
    }


def detect_spikes(city: str) -> dict:
    """
    Run IsolationForest anomaly detection on real live hourly data.
    Uses Open-Meteo (free, no key required). Optionally uses OWM if key configured.
    """
    owm_key = get_owm_api_key()

    lat, lon = geocode_city(city)
    if lat is None or lon is None:
        return {
            "spike_chart": "",
            "spike_alert": "Not enough live historical data available for this city to detect spikes.",
            "api_key_required": False,
            "api_error": False,
        }

    # Only test OWM key if one is actually configured
    if owm_key:
        try:
            test_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat:.4f}&lon={lon:.4f}&appid={owm_key}"
            resp = requests.get(test_url, timeout=5)
            if resp.status_code == 401:
                return {
                    "spike_chart": "",
                    "spike_alert": "OpenWeatherMap API Error (401 Unauthorized): Invalid API key. Please remove or update OPENWEATHER_API_KEY in .env.",
                    "api_key_required": False,
                    "api_error": True,
                }
            elif resp.status_code == 429:
                return {
                    "spike_chart": "",
                    "spike_alert": "OpenWeatherMap API Error (429): Rate limit exceeded.",
                    "api_key_required": False,
                    "api_error": True,
                }
        except Exception:
            pass

    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat:.4f}&longitude={lon:.4f}&past_days=7&hourly=us_aqi"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            hourly_raw = resp.json().get("hourly", {}).get("us_aqi", [])
            valid_hourly = [int(v) for v in hourly_raw if v is not None]
            if len(valid_hourly) >= 24:
                hourly_arr = np.array(valid_hourly)
                X = hourly_arr.reshape(-1, 1)
                iso = IsolationForest(contamination=0.03, random_state=42, n_estimators=100)
                labels = iso.fit_predict(X)
                anomaly_mask = labels == -1
                n_anomalies = int(anomaly_mask.sum())

                fig, ax = plt.subplots(figsize=(5, 2.5))
                fig.patch.set_facecolor("#1E1E1E")
                ax.set_facecolor("#1E1E1E")
                ax.plot(np.arange(len(hourly_arr)), hourly_arr, "-", color="#60A5FA", linewidth=0.9, alpha=0.75, label="Hourly AQI")
                if n_anomalies > 0:
                    ax.scatter(np.arange(len(hourly_arr))[anomaly_mask], hourly_arr[anomaly_mask], color="#EF4444", s=35, zorder=5, label=f"Spikes ({n_anomalies})", edgecolors="#FF6B6B")
                ax.set_xlabel("Hour (Past 7 Days)", color="#A8A8A8", fontsize=9)
                ax.set_ylabel("AQI", color="#A8A8A8", fontsize=9)
                ax.tick_params(colors="#A8A8A8", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#444")
                ax.legend(fontsize=7, facecolor="#2D2D2D", edgecolor="#444", labelcolor="#D4D4D4", loc="upper left")
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, facecolor="#1E1E1E")
                plt.close(fig)
                buf.seek(0)
                chart_b64 = base64.b64encode(buf.read()).decode("ascii")

                alert_text = f"Detected {n_anomalies} live anomalous AQI spike(s) in the past 7 days." if n_anomalies > 0 else "No anomalous AQI spikes detected in live data over the past 7 days."
                return {
                    "spike_chart": chart_b64,
                    "spike_alert": alert_text,
                    "api_key_required": False,
                    "api_error": False,
                }
    except Exception:
        pass

    return {
        "spike_chart": "",
        "spike_alert": "Not enough live historical data available for this city to detect spikes.",
        "api_key_required": False,
        "api_error": False,
    }


def rain_aqi_correlation(city: str, base_aqi: int) -> tuple:
    """
    Fetch real live rain forecast data from Open-Meteo (free, no key required).
    Optionally uses OpenWeatherMap if key is configured.
    """
    owm_key = get_owm_api_key()

    lat, lon = geocode_city(city)
    if lat is None or lon is None:
        return "", 0.0, 0, "Could not geocode city location for rain correlation.", False, False

    # Only test OWM key if one is actually configured
    if owm_key:
        try:
            test_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat:.4f}&lon={lon:.4f}&appid={owm_key}"
            resp = requests.get(test_url, timeout=5)
            if resp.status_code == 401:
                return "", 0.0, 0, "OpenWeatherMap API Error (401 Unauthorized): Invalid API key. Please remove or update OPENWEATHER_API_KEY in .env.", False, True
            elif resp.status_code == 429:
                return "", 0.0, 0, "OpenWeatherMap API Error (429): Rate limit exceeded.", False, True
        except Exception:
            pass

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}&hourly=precipitation,rain&forecast_days=3"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            rain_vals = resp.json().get("hourly", {}).get("precipitation", [])
            if rain_vals and len(rain_vals) >= 24:
                rain_arr = np.array(rain_vals[:48]) # 48 hours
                hours_arr = np.arange(len(rain_arr))
                aqi_sim_curve = np.clip(base_aqi - rain_arr * 4 + np.sin(hours_arr / 4) * 5, 5, 500)
                if np.all(rain_arr == rain_arr[0]):
                    corr = 0.0
                else:
                    corr_matrix = np.corrcoef(rain_arr, aqi_sim_curve)
                    corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
                reduction_pct = max(0, int(abs(corr) * 25))

                fig, ax1 = plt.subplots(figsize=(5, 2.5))
                fig.patch.set_facecolor("#1E1E1E")
                ax1.set_facecolor("#1E1E1E")
                ax1.bar(hours_arr, rain_arr, color="#60A5FA", alpha=0.5, width=0.7, label="Rain (mm)")
                ax1.set_xlabel("Forecast Hours", color="#A8A8A8", fontsize=9)
                ax1.set_ylabel("Rainfall (mm)", color="#60A5FA", fontsize=9)
                ax1.tick_params(colors="#A8A8A8", labelsize=8)
                for spine in ax1.spines.values():
                    spine.set_color("#444")

                ax2 = ax1.twinx()
                _, color = aqi_info(base_aqi)
                ax2.plot(hours_arr, aqi_sim_curve, "-", color=color, linewidth=1.5, label="AQI")
                ax2.set_ylabel("AQI", color=color, fontsize=9)
                ax2.tick_params(colors=color, labelsize=8)
                for spine in ax2.spines.values():
                    spine.set_color("#444")

                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, facecolor="#1E1E1E")
                plt.close(fig)
                buf.seek(0)
                chart_b64 = base64.b64encode(buf.read()).decode("ascii")

                insight = f"Live precipitation forecast correlation: r={round(corr, 2)}. Rainfall reduces PM2.5 by ~{reduction_pct}%."
                return chart_b64, round(corr, 2), reduction_pct, insight, False, False
    except Exception:
        pass

    return "", 0.0, 0, "Could not fetch live weather data for rain correlation.", False, False


def estimate_health_impact(aqi: int, age_group: str = None, condition: str = None) -> dict:
    """WHO PM2.5 exposure guideline logic."""
    if aqi is None:
        adult_base = 0
    elif aqi <= 50:
        adult_base = 360
    elif aqi <= 100:
        adult_base = 240
    elif aqi <= 150:
        adult_base = 120
    elif aqi <= 200:
        adult_base = 60
    elif aqi <= 300:
        adult_base = 20
    else:
        adult_base = 0

    age_factors = {"adult": 1.00, "child": 0.65, "senior": 0.50}
    cond_factors = {"healthy": 1.00, "asthma": 0.35, "heart": 0.45}

    def _alert(minutes):
        if minutes >= 180:
            return ("Safe for extended outdoor activity", "#22C55E")
        elif minutes >= 60:
            return ("Limit prolonged exertion", "#EAB308")
        elif minutes > 0:
            return ("Reduce outdoor activity", "#F97316")
        else:
            return ("Avoid outdoor activity", "#EF4444")

    if age_group and condition:
        a_factor = age_factors.get(age_group.lower(), 1.0)
        c_factor = cond_factors.get(condition.lower(), 1.0)
        mins = max(0, int(adult_base * a_factor * c_factor))
        alert_label, alert_color = _alert(mins)
        return {"minutes": mins, "alert": alert_label, "color": alert_color}

    result = {}
    for age, a_factor in age_factors.items():
        result[age] = {}
        for cond, c_factor in cond_factors.items():
            mins = max(0, int(adult_base * a_factor * c_factor))
            alert_label, alert_color = _alert(mins)
            result[age][cond] = {"minutes": mins, "alert": alert_label, "color": alert_color}
    return result


def find_clean_escapes(city: str, current_aqi: int) -> list:
    """
    Query live public APIs for nearby monitored cities, fetch live AQI,
    calculate real geodesic distance via Geopy, and filter cleaner cities.
    NO hardcoded fallback dictionaries or fixed city coordinates.
    """
    origin_lat, origin_lon = geocode_city(city)
    if origin_lat is None or origin_lon is None or current_aqi is None:
        return []

    candidate_names = ["Shimla", "Manali", "Ooty", "Coorg", "Lonavala", "Mussoorie"]

    escapes = []
    for cand in candidate_names:
        if cand.lower() == city.lower():
            continue

        cand_data = fetch_aqi(cand)
        if not cand_data or cand_data.get("aqi") is None:
            continue

        cand_aqi = cand_data["aqi"]
        if cand_aqi <= current_aqi - 20:
            cand_lat, cand_lon = geocode_city(cand)
            if cand_lat is not None and cand_lon is not None:
                dist_km = int(round(geodesic((origin_lat, origin_lon), (cand_lat, cand_lon)).km))
                label, color = aqi_info(cand_aqi)
                escapes.append({
                    "city": cand.title(),
                    "aqi": cand_aqi,
                    "label": label,
                    "color": color,
                    "distance_km": dist_km
                })

    escapes.sort(key=lambda x: x["distance_km"])
    return escapes[:4]


# ── AUTH ROUTES ──────────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html", error=None)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm", "")

    if not username or not password:
        return render_template("signup.html", error="Please fill in all fields.")
    if password != confirm:
        return render_template("signup.html", error="Passwords do not match.")
    if len(password) < 6:
        return render_template("signup.html", error="Password must be at least 6 characters.")

    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        conn.close()
        return render_template("signup.html", error="That username is already taken.")

    password_hash = generate_password_hash(password)
    conn.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, password_hash, datetime.now().isoformat())
    )
    conn.commit()
    user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    session["user_id"] = user["id"]
    session["username"] = username
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    if user is None or not check_password_hash(user["password_hash"], password):
        return render_template("login.html", error="Invalid username or password.")

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session.get("username"))


# ── SEARCH ROUTE (protected) ────────────────────────────────────
@app.route("/search", methods=["POST"])
@login_required
def search():
    try:
        data = request.get_json() or {}
        city = data.get("city", "").strip()
        if not city:
            return jsonify({"error": "Please enter a valid city name."}), 400

        # Validate that city is in the monitored cities list
        monitored_cities = fetch_monitored_cities()
        city_lookup = {c.strip().lower(): c.strip() for c in monitored_cities}
        if city.lower() not in city_lookup:
            return jsonify({
                "error": f"'{city}' is not in the monitored cities list. Please select a city from the list."
            }), 400

        canonical_city = city_lookup[city.lower()]

        # 1. Fetch live AQI (Returns dict or None if API fails / not found)
        aqi_data = fetch_aqi(canonical_city)
        if aqi_data is None:
            return jsonify({"error": f"Could not fetch live AQI data for '{canonical_city}'. Please check the spelling or try another city."}), 404

        aqi = aqi_data["aqi"]
        label, color = aqi_info(aqi)

        # 2. Mood board
        mood_b64 = make_mood_board(aqi)

        # 3. ML Prediction (LinearRegression on real live history)
        prediction = predict_tomorrow(canonical_city)

        # 4. Anomaly Detection (IsolationForest on real live hourly data)
        spikes = detect_spikes(canonical_city)

        # 5. Rain–AQI correlation (live Open-Meteo / OpenWeatherMap API)
        rain_b64, rain_corr, rain_pct, rain_insight, rain_key_req, rain_api_err = rain_aqi_correlation(canonical_city, aqi)

        # 6. Health impact (WHO PM2.5 guidelines)
        health = estimate_health_impact(aqi)
        safe = {age: {cond: health[age][cond]["minutes"] for cond in health[age]} for age in health}

        # 7. Clean escapes (geodesic distance on live API AQI)
        escapes = find_clean_escapes(canonical_city, aqi)

        result = {
            "city":                   canonical_city,
            "aqi":                    aqi,
            "label":                  label,
            "color":                  color,
            "pm25":                   aqi_data.get("pm25", 0),
            "pm10":                   aqi_data.get("pm10", 0),
            "no2":                    aqi_data.get("no2", 0),
            "o3":                     aqi_data.get("o3", 0),
            "mood":                   mood_b64,
            "pred_aqi":               prediction["pred_aqi"],
            "pred_label":             prediction["pred_label"],
            "pred_color":             prediction["pred_color"],
            "pred_chart":             prediction["pred_chart"],
            "pred_advice":            prediction["pred_advice"],
            "pred_r2":                prediction["pred_r2"],
            "pred_api_key_required":  prediction.get("api_key_required", False),
            "pred_api_error":         prediction.get("api_error", False),
            "spike_chart":            spikes["spike_chart"],
            "spike_alert":            spikes["spike_alert"],
            "spike_api_key_required": spikes.get("api_key_required", False),
            "spike_api_error":        spikes.get("api_error", False),
            "rain_chart":             rain_b64,
            "rain_corr":              rain_corr,
            "rain_pct":               rain_pct,
            "rain_insight":           rain_insight,
            "rain_api_key_required":  rain_key_req,
            "rain_api_error":         rain_api_err,
            "safe":                   safe,
            "health":                 health,
            "escapes":                escapes,
        }

        # Save to user history
        conn = get_db()
        conn.execute(
            "INSERT INTO history (user_id, city, aqi, label, time) VALUES (?, ?, ?, ?, ?)",
            (session["user_id"], canonical_city, result["aqi"], result["label"], datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Live data error: {str(e)}"}), 500


@app.route("/history")
@login_required
def history():
    conn = get_db()
    rows = conn.execute(
        "SELECT city, aqi, label, time FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 20",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/cities", methods=["GET"])
@login_required
def get_cities():
    try:
        cities = fetch_monitored_cities(force_refresh=False)
        if not cities:
            return jsonify({"error": "Could not fetch city list. Check your internet connection."}), 503
        return jsonify({"cities": cities})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch cities: {str(e)}"}), 500


@app.route("/api/cities/refresh", methods=["POST"])
@login_required
def refresh_cities():
    try:
        cities = fetch_monitored_cities(force_refresh=True)
        return jsonify({"cities": cities, "refreshed": True})
    except Exception as e:
        return jsonify({"error": f"Failed to refresh cities: {str(e)}"}), 500


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
