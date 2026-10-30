"""
confusion_matrix.py - classification-style accuracy of the Random Forest forecast.

The forest predicts a NUMBER (load ratio = cars / road capacity).  To get a confusion matrix we
turn that number into three classes, using the same colours as the map:

    Free   : load < 35 %
    Busy   : 35 % - 65 %
    Jammed : load > 65 %

    python confusion_matrix.py          # about 1 minute, writes confusion_matrix.json

It uses the SAME held-out episodes as train_model.py (the last 20 %), which the forest never saw
while training, and compares the forest with the simple "same as now" baseline.

Why per-class recall and precision matter: almost every road is Free (about 96 % of road-hours),
so overall accuracy looks high even for a useless model.  Recall tells you how many of the
Busy / Jammed roads were really found; precision tells you how many alarms were real.
"""
import json

import joblib
import numpy as np

import train_model as T
from forecast_model import FEATURES, MODEL_PATH

EDGES = [0.35, 0.65]
NAMES = ["Free", "Busy", "Jammed"]
HORIZONS = {"+1h": 0, "+2h": 1, "+3h": 2, "+4h": 3}


def classify(load):
    return np.digitize(load, EDGES)


def matrix(actual, predicted):
    m = np.zeros((3, 3), dtype=int)
    for a, p in zip(actual, predicted):
        m[a, p] += 1
    return m


def summarise(m):
    recall = [m[i, i] / m[i].sum() if m[i].sum() else 0.0 for i in range(3)]
    precision = [m[i, i] / m[:, i].sum() if m[:, i].sum() else 0.0 for i in range(3)]
    f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(precision, recall)]
    return dict(matrix=m.tolist(), accuracy=float(np.trace(m) / m.sum()),
                recall=[round(v, 4) for v in recall], precision=[round(v, 4) for v in precision],
                macro_f1=round(float(np.mean(f1)), 4))


def main():
    blob = joblib.load(MODEL_PATH)
    rf = blob["model"]
    rf.n_jobs = 1
    episodes, hours = blob["metrics"]["episodes"], blob["metrics"]["hours"]
    n_test = max(1, episodes // 5)
    print(f"Re-simulating the {n_test} held-out episodes ...", flush=True)
    res = [T.run_episode((s, hours)) for s in range(episodes - n_test + 1, episodes + 1)]
    X = np.concatenate([r[0] for r in res])
    Y = np.concatenate([r[1] for r in res])
    forest = rf.predict(X)
    same_now = np.repeat(X[:, FEATURES.index("load")][:, None], 4, axis=1)

    out = dict(classes=NAMES, thresholds=EDGES, n_samples=int(len(X)), models={})
    for name, pred in (("random_forest", forest), ("same_as_now", same_now)):
        out["models"][name] = {h: summarise(matrix(classify(Y[:, k]), classify(pred[:, k]))) for h, k in HORIZONS.items()}
    share = np.bincount(classify(Y[:, 3]), minlength=3) / len(Y)
    out["class_share"] = [round(float(v), 4) for v in share]
    out["always_free_accuracy"] = round(float(share[0]), 4)

    with open("confusion_matrix.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nTest samples: {len(X):,}   class share Free/Busy/Jammed: {out['class_share']}")
    print(f"Accuracy of always guessing 'Free': {out['always_free_accuracy']:.3f}")
    for h in ("+1h", "+4h"):
        for name in ("random_forest", "same_as_now"):
            r = out["models"][name][h]
            print(f"\n{h} {name}: accuracy {r['accuracy']:.3f}   macro-F1 {r['macro_f1']:.3f}")
            print("   actual\\pred   " + "  ".join(f"{n:>7}" for n in NAMES) + "   recall  precision")
            for i, n in enumerate(NAMES):
                print(f"   {n:<11}  " + "  ".join(f"{v:7d}" for v in r["matrix"][i]) + f"   {r['recall'][i]:6.3f}  {r['precision'][i]:6.3f}")


if __name__ == "__main__":
    main()
