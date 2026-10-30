"""
train_model.py - builds the training data and trains the Random Forest.

    python train_model.py                 # default: 40 simulated episodes of 72 hours
    python train_model.py --episodes 60   # more data = better model, longer training

What happens
  1. The traffic engine is run headlessly many times ("episodes").  Each episode has a random
     demand level, a random response mode and 8-14 random accidents (junction or road,
     partial or full closure, 1-8 hours long).
  2. Every simulated hour we record the features of every road (forecast_model.build_features)
     and the real load of that road 1, 2, 3 and 4 hours later (= the targets).
  3. A RandomForestRegressor learns  features -> load at +1h, +2h, +3h, +4h.
  4. The last episodes are held out as a test set; the model is compared against the simplest
     possible baseline ("load stays what it is now").
"""
import os
import sys
import json
import time
import random
import argparse
from multiprocessing import Pool

import numpy as np

import engine as E
from forecast_model import build_features, FEATURES, MODEL_PATH

HORIZONS = 4


def run_episode(args):
    seed, hours = args
    rng = random.Random(seed * 7919 + 13)
    net = E.Network()
    eng = E.Engine(net, seed=seed, demand=rng.uniform(0.7, 1.4),
                   mode=rng.choice(["reactive", "early"]), forecaster=None, use_rf=False)
    # random accidents spread over the episode
    n_acc = rng.randint(8, 14) * max(1, hours // 72)
    for _ in range(n_acc):
        start = eng.t + rng.uniform(1, hours - 10)
        if rng.random() < 0.5:
            kind, target = "node", rng.choice([n for n in net.nodes if net.nodes[n]["kind"] == "junction"])
        else:
            kind, target = "road", rng.randrange(len(net.roads))
        eng.add_accident(kind, target, start, rng.uniform(1, 8), rng.choice([0.5, 0.8, 1.0]))
    feats, loads = [], []
    eng.hour_hooks.append(lambda e: (feats.append(build_features(e)), loads.append(e.loads())))
    eng.run(hours)
    X, Y = [], []
    for h in range(0, len(feats) - HORIZONS):
        X.append(feats[h])
        Y.append(np.stack([loads[h + k] for k in range(1, HORIZONS + 1)], axis=1))
    return np.concatenate(X), np.concatenate(Y)


def train(episodes=40, hours=72, out=MODEL_PATH, workers=None, verbose=True):
    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestRegressor

    t0 = time.time()
    jobs = [(s, hours) for s in range(1, episodes + 1)]
    workers = workers or max(1, min(os.cpu_count() or 1, episodes))
    if verbose:
        print(f"Simulating {episodes} episodes x {hours} h on {workers} process(es) ...")
    if workers > 1:
        with Pool(workers) as p:
            results = p.map(run_episode, jobs)
    else:
        results = [run_episode(j) for j in jobs]

    n_test = max(1, episodes // 5)
    Xtr = np.concatenate([r[0] for r in results[:-n_test]])
    Ytr = np.concatenate([r[1] for r in results[:-n_test]])
    Xte = np.concatenate([r[0] for r in results[-n_test:]])
    Yte = np.concatenate([r[1] for r in results[-n_test:]])
    if verbose:
        print(f"Training samples: {len(Xtr):,}   test samples: {len(Xte):,}   features: {len(FEATURES)}")

    # roads close to a known accident are rare, so give those samples extra weight
    near_tr = Xtr[:, FEATURES.index("acc_dist")] <= 2
    weight = np.where(near_tr, 4.0, 1.0)
    rf = RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=10, max_features=0.7,
                               n_jobs=-1, random_state=0)
    rf.fit(Xtr, Ytr, sample_weight=weight)

    pred = rf.predict(Xte)
    base = np.repeat(Xte[:, FEATURES.index("load")][:, None], HORIZONS, axis=1)   # persistence baseline
    near = (Xte[:, FEATURES.index("acc_dist")] <= 2) & (Xte[:, FEATURES.index("acc_dist")] < 7)
    # roads that lead INTO an accident that is happening right now (where queues build up)
    hot = ((Xte[:, FEATURES.index("acc_active")] == 1) & (Xte[:, FEATURES.index("acc_toward")] == 1)
           & (Xte[:, FEATURES.index("acc_dist")] == 0))
    metrics = dict(
        trained_at=time.strftime("%Y-%m-%d %H:%M"),
        n_train=int(len(Xtr)), n_test=int(len(Xte)), n_trees=60, episodes=episodes, hours=hours,
        horizons=[f"+{h}h" for h in range(1, HORIZONS + 1)],
        mae_rf=[round(float(np.abs(pred[:, k] - Yte[:, k]).mean()), 4) for k in range(HORIZONS)],
        mae_baseline=[round(float(np.abs(base[:, k] - Yte[:, k]).mean()), 4) for k in range(HORIZONS)],
        mae_rf_near_accident=[round(float(np.abs(pred[near, k] - Yte[near, k]).mean()), 4) for k in range(HORIZONS)] if near.any() else None,
        mae_baseline_near_accident=[round(float(np.abs(base[near, k] - Yte[near, k]).mean()), 4) for k in range(HORIZONS)] if near.any() else None,
        n_test_active_roads=int(hot.sum()),
        mae_rf_active_roads=[round(float(np.abs(pred[hot, k] - Yte[hot, k]).mean()), 4) for k in range(HORIZONS)] if hot.any() else None,
        mae_baseline_active_roads=[round(float(np.abs(base[hot, k] - Yte[hot, k]).mean()), 4) for k in range(HORIZONS)] if hot.any() else None,
        feature_importance={f: round(float(v), 3) for f, v in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])},
        note="MAE is in 'load ratio' units (cars on road / road capacity); lower is better.",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(dict(model=rf, features=FEATURES, metrics=metrics, sklearn=sklearn.__version__), out, compress=3)
    with open(out.replace(".joblib", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print("\nHold-out MAE (load ratio, lower = better)")
        print("  horizon          ", "  ".join(f"{h:>7}" for h in metrics["horizons"]))
        print("  Random Forest    ", "  ".join(f"{v:7.4f}" for v in metrics["mae_rf"]))
        print("  'same as now'    ", "  ".join(f"{v:7.4f}" for v in metrics["mae_baseline"]))
        if metrics["mae_rf_near_accident"]:
            print("  RF near accident ", "  ".join(f"{v:7.4f}" for v in metrics["mae_rf_near_accident"]))
            print("  base near accid. ", "  ".join(f"{v:7.4f}" for v in metrics["mae_baseline_near_accident"]))
        if metrics["mae_rf_active_roads"]:
            print("  RF, roads into active accident ", "  ".join(f"{v:7.4f}" for v in metrics["mae_rf_active_roads"]))
            print("  base, same roads               ", "  ".join(f"{v:7.4f}" for v in metrics["mae_baseline_active_roads"]))
        print("  Top features:", ", ".join(list(metrics["feature_importance"])[:5]))
        print(f"\nSaved {out}  ({os.path.getsize(out) / 1e6:.1f} MB) in {time.time() - t0:.0f} s")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--hours", type=int, default=72)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    train(a.episodes, a.hours, workers=a.workers)
