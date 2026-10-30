"""
report_data.py - re-creates the numbers shown in the presentation graphs.

    python report_data.py            # writes report_data.json (about 1 minute)

It runs the same engine.py and the trained Random Forest as the web app, with fixed random
seeds, so every run gives the same numbers.

Scenarios (all start at 08:00, seed 11)
  A  no accident, 24 h                     -> cars on the map, cars entering / leaving per hour
  B  junction B2 fully closed 13:00-19:00  -> run twice: drivers 'reactive' and 'early'
                                             (stuck cars over time, delay, RF forecast vs reality)
"""
import json
import sys

import engine as E
import forecast_model as F

SEED = 11
HOURS = 24
ACC = dict(kind="node", target="B2", start_in=5.0, duration=6.0, severity=1.0)


def make(mode, accident, forecaster):
    net = E.Network()
    eng = E.Engine(net, seed=SEED, forecaster=forecaster, mode=mode, use_rf=True)
    if accident:
        eng.add_accident(ACC["kind"], ACC["target"], eng.t + ACC["start_in"], ACC["duration"], ACC["severity"])
    return net, eng


def stuck_now(eng):
    return sum(1 for vs in eng.veh for c in vs if c.stuck)


def run(mode, accident, forecaster, watch_edges=()):
    net, eng = make(mode, accident, forecaster)
    t0 = eng.t
    rec = dict(t=[], stuck=[], rerouted=[], cars=[], hour_t=[], hour_cars=[], entered=[], exited=[],
               watch_actual={}, watch_pred4={})
    for ei in watch_edges:
        rec["watch_actual"][ei] = {}
        rec["watch_pred4"][ei] = {}
    last = dict(entered=0, exited=0)

    def hook(e):                                           # called at every full simulated hour
        h = round(e.t)
        rec["hour_t"].append(h)
        rec["hour_cars"].append(sum(len(v) for v in e.veh))
        rec["entered"].append(e.entered - last["entered"])
        rec["exited"].append(e.exited - last["exited"])
        last["entered"], last["exited"] = e.entered, e.exited
        ld = e.loads()
        for ei in watch_edges:
            rec["watch_actual"][ei][h] = ld[ei]
            if e.pred is not None:
                rec["watch_pred4"][ei][h + 4] = float(e.pred[ei][3])     # forecast for 4 hours later
    eng.hour_hooks.append(hook)
    next_sample = 0.5                                      # record every half hour of simulated time
    while eng.t - t0 < HOURS - 1e-6:
        eng.step()
        if eng.t - t0 >= next_sample - 1e-6:
            rec["t"].append(next_sample)
            rec["stuck"].append(stuck_now(eng))
            rec["rerouted"].append(eng.rerouted_total)
            rec["cars"].append(sum(len(v) for v in eng.veh))
            next_sample += 0.5
    rec["avg_delay_min"] = round(sum(eng.delays) / len(eng.delays) * 60, 1) if eng.delays else 0.0
    rec["peak_stuck"] = max(rec["stuck"])
    rec["rerouted_total"] = eng.rerouted_total
    rec["accidents"] = [dict(label=a["label"], n_stuck=a["n_stuck"], n_rerouted=a["n_rerouted"]) for a in eng.accidents]
    return net, eng, rec


def main():
    fc = F.Forecaster()
    net = E.Network()
    watch = list(net.inc["B2"])
    print("A: no accident ...", flush=True)
    _, _, none = run("reactive", False, fc, watch)
    print("B1: accident, drivers react when it happens ...", flush=True)
    _, _, react = run("reactive", True, fc, watch)
    print("B2: accident, drivers act on the 4-hour warning ...", flush=True)
    _, _, early = run("early", True, fc, watch)

    # which road into B2 got the most crowded (reactive run)?
    def peak(ei):
        return max(react["watch_actual"][ei].values())
    best = max(watch, key=peak)
    hours = sorted(react["watch_actual"][best])
    series = []
    for h in hours:
        if h in react["watch_pred4"][best] or h + 0 in react["watch_actual"][best]:
            series.append(h)
    window = [h for h in range(8 + 4, 8 + HOURS + 1) if h in react["watch_actual"][best] and h in react["watch_pred4"][best]
              and (h - 4) in react["watch_actual"][best]]
    actual = [react["watch_actual"][best][h] for h in window]
    rf = [react["watch_pred4"][best][h] for h in window]
    naive = [react["watch_actual"][best][h - 4] for h in window]
    mae = lambda a, b: sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    out = dict(
        seed=SEED, accident=dict(ACC, label="B2 Central Junction", start_clock="13:00", end_clock="19:00", warning_clock="09:00"),
        road=net.edge_name(best), window_hours=window,
        forecast_vs_actual=dict(actual=actual, rf=rf, naive=naive,
                                mae_rf=mae(actual, rf), mae_naive=mae(actual, naive)),
        none=none, reactive=react, early=early,
    )
    # keep the json small
    for r in (none, react, early):
        r.pop("watch_actual"); r.pop("watch_pred4")
    with open("report_data.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nroad", out["road"], "  MAE RF", round(out["forecast_vs_actual"]["mae_rf"], 4),
          " MAE same-as-4h-ago", round(out["forecast_vs_actual"]["mae_naive"], 4))
    for k, r in (("no accident", none), ("react", react), ("early", early)):
        print(f"{k:12s} peak stuck {r['peak_stuck']:4d}  rerouted {r['rerouted_total']:4d}  avg delay {r['avg_delay_min']:6.1f} min")


if __name__ == "__main__":
    main()
