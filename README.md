🚦 Traffic Flow Prediction & 4-Hour Congestion Early Warning System
A live traffic simulation that uses a Random Forest model to forecast road congestion 1 to 4 hours ahead and issues an early warning 4 hours before a scheduled accident, so drivers can be rerouted before they get stuck in the queue.

## 📌 Problem Statement

Most navigation systems react to a jam only after it has formed. By then, hundreds of vehicles are already committed to the blocked roads. This project explores what happens when traffic is *predicted in advance* and drivers are warned early, and compares it with the usual "react when it happens" approach.

## ✨ Features

- *City road network*: 16 named junctions (e.g., Central Junction, Rail Station, Hospital) with diagonal shortcuts, a river with no bridge, and 8 entry/exit gateways
- *Continuous traffic*: cars keep entering from one gateway and leaving through another, following a realistic morning and evening rush-hour demand profile
- *Queue-based road model*: each road has a capacity, so jams build up and spill back into upstream roads
- *Accident scheduling*: add any number of accidents on junctions or roads, each with its own start time, duration and severity (partial or full closure)
- *4-hour early warning*: a warning is issued 4 hours before each accident starts
- *Random Forest forecast*: predicts the load of every road at +1 h, +2 h, +3 h and +4 h
- *Smart rerouting*: Dijkstra routing that uses the forecast and avoids known accidents
- *Two driver-response modes*
  - React when accident happens: cars less than 4 h from the accident get stuck, cars further away are rerouted
  - Act on the 4-hour warning: cars heading for the accident are rerouted as soon as the warning is issued
- *Live dashboard*: clock, cars on map, entered/left, stuck, rerouted, average delay per trip, event log and colour-coded roads (load now or forecast)

## 🧠 How It Works


 Traffic engine (engine.py)  ──►  Road features every hour  ──►  Random Forest
   cars, queues, accidents          (18 per road)                  (+1h … +4h load)
            ▲                                                          │
            └────────────── forecast-aware rerouting ◄─────────────────┘
                                        │
                              Flask API (app.py)  ──►  Browser map (canvas)


1. *Simulation*: engine.py moves every car in steps of 1.2 minutes of simulated time.
2. *Training data*: train_model.py runs the engine headlessly for 40 episodes × 72 hours with random demand and 8–14 random accidents each.
3. *Features (18 per road)*: time of day, current load and the last 3 hours of load, upstream and downstream load, share of stopped cars, road length and capacity, and details of the nearest known accident (distance, direction, time to start, active, severity, hours left), plus the driver-response mode.
4. *Model*: one RandomForestRegressor (60 trees, max depth 14) with 4 outputs, one for each forecast hour. Samples near accidents get 4× weight because they are rare.
5. *Routing*: the forecast feeds into the travel-time estimate used by Dijkstra's algorithm.

## 📊 Results

Hold-out test on simulated data (145,728 training samples, 36,432 test samples). Error is the Mean Absolute Error of the *load ratio* (cars on road ÷ road capacity). Lower is better.

| Forecast horizon | Random Forest | Baseline ("same as now") | Improvement |
|---|---|---|---|
| +1 h | 0.0439 | 0.0550 | ~20% |
| +2 h | 0.0535 | 0.0774 | ~31% |
| +3 h | 0.0599 | 0.0892 | ~33% |
| +4 h | 0.0627 | 0.0974 | ~36% |

*Effect of the early warning* (Central Junction B2 fully closed 13:00–19:00, same random seed):

| Scenario | Peak cars stuck | Cars rerouted | Avg delay per trip |
|---|---|---|---|
| No accident | 0 | 0 | 20.8 min |
| React when accident happens | 67 | 8 | 37.7 min |
| Act on the 4-hour warning | *3* | *117* | *23.2 min* |

Acting on the warning cut the peak number of stuck cars from 67 to 3 and brought the average delay close to the no-accident level.

The most important features were current load (35%) and upstream load (23%), followed by time of day.

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Simulation & ML | Python, NumPy, scikit-learn (RandomForestRegressor), joblib |
| Backend | Flask (REST API) |
| Frontend | HTML, CSS, JavaScript (Canvas) |

## 📁 Project Structure


traffic_4hour_warning/
├── app.py                  # Flask server and REST API
├── engine.py               # Traffic world: roads, queues, cars, accidents, routing
├── forecast_model.py       # Feature builder + Random Forest loader
├── train_model.py          # Simulates training data, trains and evaluates the model
├── confusion_matrix.py     # Free / Busy / Jammed confusion matrices, recall, precision
├── report_data.py          # Re-creates the numbers used in the report graphs
├── models/
│   ├── rf_traffic.joblib           # Trained Random Forest
│   └── rf_traffic_metrics.json     # Hold-out error and feature importance
├── templates/index.html    # Web interface
├── static/app.js           # Map drawing and controls
├── static/style.css        # Styling
└── requirements.txt


## 🚀 Getting Started

*Requirements:* Python 3.10 or newer

bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>/traffic_4hour_warning

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Re-train the model (about 1 minute)
python train_model.py

# 5. Run the app
python app.py


Open *http://127.0.0.1:5000* in your browser.

> A trained model is included. If it is missing or was made with a different scikit-learn version, app.py re-trains it automatically on start-up.

## 🎮 How to Use

1. Click a *junction* or a *road* on the map.
2. Set Starts in, Lasts and Severity, then press *Add accident* (or *Random accident*).
3. Press *▶️ Start*. Change speed, traffic demand and driver response at any time.
4. Colour the roads by *Load now* or by the Random Forest forecast for *+1 h … +4 h*.
5. Watch the event log for the early warning, stuck and rerouted cars.

### Extra scripts

bash
python confusion_matrix.py   # classification view of the forecast (Free / Busy / Jammed)
python report_data.py        # reproduce the numbers in the report graphs


## ⚠️ Limitations

- The model is trained and tested on *simulated* data, not real city traffic.
- Accidents are *scheduled by the user, so the system knows them in advance. The model learns how traffic *reacts to a known accident; it does not predict that an accident will happen.
- About 96% of road-hours are free-flowing, so overall accuracy looks high. Recall for Busy and Jammed roads drops at longer horizons, and very large queues are under-predicted.
- Accident-related features carry only about 7% of the model's importance.

## 🔮 Future Scope

- Train on real traffic sensor and incident data (for example, PeMS or city open data)
- Try gradient boosting or LSTM models for longer horizons
- Connect to a real map (OpenStreetMap) instead of a grid
- Send warnings to drivers through a mobile app or SMS

## 📄 License

This project was developed as an academic micro-project for the Machine Learning course. Feel free to use it for learning purposes.
