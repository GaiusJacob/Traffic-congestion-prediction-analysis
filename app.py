"""
app.py - Flask server.

The simulation (engine.py) and the Random Forest (forecast_model.py) run here in Python.
The browser (static/app.js) asks for the latest state ~8 times a second and draws it.
"""
import os
import sys
import random
import subprocess
import threading

from flask import Flask, jsonify, render_template, request

import engine as E
import forecast_model as F

app = Flask(__name__)
NET = E.Network()
LOCK = threading.Lock()
FORECASTER = None
MODEL_MESSAGE = ""
SETTINGS = dict(demand=1.0, mode="reactive", use_rf=True)
ENGINE = None


def load_forecaster():
    """Load the trained Random Forest; train it first if it does not exist yet."""
    global FORECASTER, MODEL_MESSAGE
    if not os.path.exists(F.MODEL_PATH):
        print("No trained model found - training one now (about 30-60 s) ...", flush=True)
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "train_model.py")], check=False)
    for attempt in range(2):
        try:
            FORECASTER = F.Forecaster()
            MODEL_MESSAGE = "Random Forest loaded."
            return
        except Exception as ex:                      # missing file, or scikit-learn version changed
            print(f"Could not load model ({ex}). Retraining ...", flush=True)
            if attempt == 0:
                subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "train_model.py")], check=False)
    FORECASTER = None
    MODEL_MESSAGE = "Random Forest could not be loaded - simulation runs without forecasts."


def new_engine():
    global ENGINE
    ENGINE = E.Engine(NET, seed=random.randrange(1, 10**6), demand=SETTINGS["demand"], forecaster=FORECASTER,
                      mode=SETTINGS["mode"], use_rf=SETTINGS["use_rf"])
    return ENGINE


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/network")
def network():
    return jsonify(NET.to_json())


@app.get("/api/state")
def state():
    with LOCK:
        return jsonify(ENGINE.state(int(request.args.get("since", 0))))


@app.post("/api/advance")
def advance():
    data = request.get_json(silent=True) or {}
    hours = min(max(float(data.get("hours", 0.1)), 0.0), 1.0)
    with LOCK:
        if hours > 0:
            ENGINE.advance(hours)
        return jsonify(ENGINE.state(int(data.get("since", 0))))


@app.post("/api/accidents")
def add_accident():
    d = request.get_json(silent=True) or {}
    try:
        with LOCK:
            ENGINE.add_accident(d.get("kind"), d.get("target"), ENGINE.t + float(d.get("start_in", 5)),
                                float(d.get("duration", 6)), float(d.get("severity", 1.0)))
            return jsonify(ENGINE.state(int(d.get("since", 0))))
    except (ValueError, TypeError) as ex:
        return jsonify(error=str(ex)), 400


@app.post("/api/accidents/random")
def random_accident():
    d = request.get_json(silent=True) or {}
    with LOCK:
        if random.random() < 0.5:
            kind, target = "node", random.choice([n for n in NET.nodes if NET.nodes[n]["kind"] == "junction"])
        else:
            kind, target = "road", random.randrange(len(NET.roads))
        ENGINE.add_accident(kind, target, ENGINE.t + random.uniform(4.5, 8), random.uniform(4, 8),
                            random.choice([0.5, 0.8, 1.0]))
        return jsonify(ENGINE.state(int(d.get("since", 0))))


@app.delete("/api/accidents/<int:acc_id>")
def cancel_accident(acc_id):
    with LOCK:
        ok = ENGINE.remove_accident(acc_id)
        return jsonify(ENGINE.state(int(request.args.get("since", 0)))), (200 if ok else 404)


@app.post("/api/settings")
def settings():
    d = request.get_json(silent=True) or {}
    with LOCK:
        if "demand" in d:
            SETTINGS["demand"] = ENGINE.demand = min(max(float(d["demand"]), 0.3), 2.5)
        if d.get("mode") in ("reactive", "early"):
            SETTINGS["mode"] = ENGINE.mode = d["mode"]
        if "use_rf" in d:
            SETTINGS["use_rf"] = ENGINE.use_rf = bool(d["use_rf"])
        return jsonify(ENGINE.state(int(d.get("since", 0))))


@app.post("/api/reset")
def reset():
    d = request.get_json(silent=True) or {}
    with LOCK:
        if "demand" in d:
            SETTINGS["demand"] = min(max(float(d["demand"]), 0.3), 2.5)
        new_engine()
        ENGINE.log("Network reset. Cars keep entering and leaving through the gateways. "
                   "Click the map to pick a junction or road, then schedule an accident.", "ok")
        return jsonify(ENGINE.state(0))


@app.get("/api/model")
def model_info():
    return jsonify(loaded=FORECASTER is not None, message=MODEL_MESSAGE, metrics=F.load_metrics())


@app.get("/api/health")
def health():
    return jsonify(status="ok", sim_time=ENGINE.t if ENGINE else None)


load_forecaster()
new_engine()
ENGINE.log("Simulation ready (08:00). Click the map to pick a junction or road, then schedule an accident.", "ok")

if __name__ == "__main__":
    # debug=False: the auto-reloader would start the simulation twice
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
