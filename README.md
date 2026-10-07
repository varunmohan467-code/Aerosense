# AeroSense — Smart Air Quality Dashboard

A full-stack Flask web application that provides **real-time, live-API-only** air quality monitoring, health impact estimation, ML-powered prediction, anomaly detection, and weather correlation for Indian cities.

> **Important:** This project requires an active internet connection for all features to work, since all AQI, historical, weather, and distance data comes from live public APIs (WAQI, OpenWeatherMap, Geopy/Nominatim geocoding). There are **no offline fallback dictionaries**, **no demo cities**, and **no simulated/random data** anywhere in the codebase.

---

## Features

| Feature | Live API Source | ML Model |
|---|---|---|
| **Current AQI** | WAQI / OpenWeatherMap Air Pollution / OpenAQ v3 | — |
| **AQI Mood Board** | Matplotlib abstract art based on live AQI | — |
| **Health Impact Estimator** | WHO PM2.5 exposure guidelines | — |
| **Tomorrow's AQI Prediction** | OpenWeatherMap Air Pollution History | scikit-learn `LinearRegression` |
| **Anomaly/Spike Detection** | OpenWeatherMap Air Pollution History | scikit-learn `IsolationForest` |
| **Rain vs AQI Correlation** | OpenWeatherMap Forecast API | NumPy Pearson correlation |
| **Clean Escape Finder** | WAQI live feed + Geopy Nominatim geocoding + geodesic distance | — |
| **Search History** | SQLite per-user persistence | — |

---

## Prerequisites

- **Python 3.10+**
- **Active internet connection** (all data is fetched live from public APIs)
- **Free OpenWeatherMap API key** (required for prediction, spike detection, and rain correlation)

---

## Getting a Free OpenWeatherMap API Key

1. Go to [https://openweathermap.org/api](https://openweathermap.org/api)
2. Click **Sign Up** and create a free account
3. Go to **API Keys** in your dashboard
4. Copy your API key (it may take a few minutes to activate after creation)

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd aerosense

# Install dependencies
pip install -r requirements.txt

# Set your OpenWeatherMap API key
# On Windows (PowerShell):
$env:OPENWEATHER_API_KEY = "your-api-key-here"

# On Linux/Mac:
export OPENWEATHER_API_KEY="your-api-key-here"

# Run the application
python app.py
```

The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Demo Login for Viva

For evaluators, guides, and viva examiners, a pre-configured demo account is available:

- **Username:** `viva_demo`
- **Password:** `viva_password123`

You can also sign up a new custom account anytime on the [Signup Page](http://127.0.0.1:5000/signup).

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENWEATHER_API_KEY` | **Yes** (for full features) | Free API key from OpenWeatherMap. Enables prediction, spike detection, and rain correlation. |
| `OPENAQ_API_KEY` | No | Optional OpenAQ v3 API key for additional AQI source. |
| `WAQI_API_KEY` | No | WAQI token (defaults to `"demo"` which works for most cities). |

> **Note:** Without `OPENWEATHER_API_KEY`, the app will still show live current AQI (via WAQI), mood board, and health advice, but prediction, spike detection, and rain correlation cards will display informative "API key required" messages instead of charts.

---

## Tech Stack

- **Backend:** Flask, SQLite, Werkzeug (password hashing)
- **ML:** scikit-learn (LinearRegression, IsolationForest)
- **Visualization:** Matplotlib (server-side chart generation)
- **Geocoding:** Geopy (Nominatim + geodesic distance)
- **APIs:** WAQI, OpenWeatherMap Air Pollution, OpenAQ v3
- **Frontend:** Vanilla HTML/CSS/JS (dark silver/gold theme)

---

## Project Structure

```
aerosense/
├── app.py                  # Flask backend (all live API logic)
├── requirements.txt        # Python dependencies
├── aerosense.db            # SQLite database (auto-created)
├── README.md               # This file
└── templates/
    ├── login.html          # Login page
    ├── signup.html         # Signup page
    └── dashboard.html      # Main dashboard (all card sections)
```

---

## API Error Handling

Every API call in the backend has proper error handling:

- **City not found:** Returns `404` with `"Could not fetch live AQI data for this city."`
- **API unreachable:** Returns `500` with `"Live data unavailable right now."`
- **Insufficient history:** Prediction/spike cards show `"Not enough live historical data."`
- **Missing API key:** Rain correlator shows `"Set OPENWEATHER_API_KEY to enable."`

The frontend displays a themed error banner instead of broken/blank cards.

---

## License

MCA Mini Project 2025–26 | Varun S Mohan | Guide: Athira Kurup
