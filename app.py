"""
Explainable AI-Based Cyber Threat Detection System
--------------------------------------------------
Flask backend: authentication, alert storage (SQLite), ML inference
(Random Forest on real CICIDS2017 flow features) and explainability
(SHAP TreeExplainer).

Run:  python train_model.py   (once, to create the model files)
      python app.py
"""

import json
import os
import random
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import shap
from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request, url_for)
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from werkzeug.security import check_password_hash, generate_password_hash

from features import (CLASS_NAMES, FEATURES, FEATURE_LABELS, FEATURE_NAMES,
                      describe_feature, format_value, SERVICE_PORTS)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "soc_database.db")
MODEL_PATH = os.path.join(BASE_DIR, "threat_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
DEMO_PATH = os.path.join(BASE_DIR, "demo_events.csv")

# Secrets come from environment variables; defaults are for local demo only.
SECRET_KEY = os.environ.get("SOC_SECRET_KEY", "dev-only-change-me")
ADMIN_USERNAME = os.environ.get("SOC_ADMIN_USER", "admin")
ADMIN_PASSWORD_HASH = generate_password_hash(os.environ.get("SOC_ADMIN_PASS", "admin123"))
DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

RISK_ORDER = ["Low", "Medium", "High", "Critical"]
MIN_REASON_IMPACT = 0.05   # minimum SHAP impact (probability units) to be reported as a reason
MAX_REASONS = 4
REPLAY_WEIGHTS = {0: 0.20, 1: 0.27, 2: 0.27, 3: 0.26}  # share of replayed events per class

app = Flask(__name__)
app.secret_key = SECRET_KEY

# ---------------------------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, user_id):
        self.id = str(user_id)


@login_manager.user_loader
def load_user(user_id):
    # Demo setup: a single admin account with id "1".
    return User(user_id) if user_id == "1" else None


def is_safe_redirect(target):
    """Allow only relative redirects (prevents open-redirect attacks)."""
    if not target:
        return False
    parsed = urlparse(target)
    return not parsed.netloc and not parsed.scheme


# ---------------------------------------------------------------------------
# MODEL + ARTIFACTS (created by train_model.py)
# ---------------------------------------------------------------------------
def load_artifacts():
    missing = [os.path.basename(p) for p in (MODEL_PATH, METRICS_PATH, DEMO_PATH) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing files: {', '.join(missing)}. Run 'python train_model.py' first.")
    model = joblib.load(MODEL_PATH)
    if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise SystemExit("threat_model.pkl was trained with a different feature set. Re-run 'python train_model.py'.")
    with open(METRICS_PATH, encoding="utf-8") as fh:
        metrics = json.load(fh)
    return model, metrics, pd.read_csv(DEMO_PATH)


print("Loading AI model...")
model, metrics, demo_events = load_artifacts()
explainer = shap.TreeExplainer(model)
CLASS_INDEX = {int(c): i for i, c in enumerate(model.classes_)}   # label -> column index
BASELINE = metrics["baseline"]                                     # benign medians per feature
DEMO_BY_CLASS = {c: demo_events[demo_events["actual_class"] == c].reset_index(drop=True) for c in CLASS_NAMES}
# Sorted packet rates per threat type: used to judge how intense a new event is compared with its peers.
RATE_REFERENCE = {c: np.sort(pool["Flow_Packets_per_s"].to_numpy()) for c, pool in DEMO_BY_CLASS.items() if len(pool)}

# ---------------------------------------------------------------------------
# PROFESSIONAL CONTENT (shown on the Investigation and System Overview pages)
# ---------------------------------------------------------------------------
MODEL_INFO = {
    "algorithm": "Random Forest Classifier (scikit-learn)",
    "explainer": "SHAP TreeExplainer",
    "dataset": "CICIDS2017 (Canadian Institute for Cybersecurity)",
    "records": f"{metrics['n_total']:,}",
    "accuracy": f"{metrics['accuracy'] * 100:.2f}%",
    "macro_f1": f"{metrics['report']['macro avg']['f1-score']:.3f}",
    "sampling": "Up to 40,000 rows sampled per daily capture file (Monday to Friday); invalid and duplicate flows removed; stratified 80/20 train/test split.",
    "features": [(f["name"], f["label"], f["description"]) for f in FEATURES],
}

THREAT_PLAYBOOK = {
    "Brute Force Attack": {
        "summary": "Repeated, automated login attempts aimed at guessing valid credentials. A successful attempt gives the attacker legitimate-looking access to the system.",
        "mitre": "T1110 - Brute Force (Credential Access)",
        "actions": [
            "Temporarily block or rate-limit the source IP at the firewall.",
            "Review authentication logs for any successful login from the same address.",
            "Force a password reset for targeted accounts and enable multi-factor authentication.",
            "Add account lockout thresholds and CAPTCHA to exposed login endpoints.",
        ],
    },
    "Port Scanning": {
        "summary": "Systematic probing of network ports to discover running services. Scanning is usually the reconnaissance phase that precedes a targeted attack.",
        "mitre": "T1046 - Network Service Discovery (Discovery)",
        "actions": [
            "Block the scanning source and monitor for follow-up connection attempts.",
            "Close or firewall all ports and services that are not required.",
            "Check exposed services for missing patches and weak configurations.",
            "Enable intrusion-prevention rules that throttle rapid multi-port connections.",
        ],
    },
    "Suspicious Attack": {
        "summary": "Traffic whose behaviour deviates from the normal baseline, such as denial-of-service floods, botnet activity or web attacks. It needs analyst review to confirm intent.",
        "mitre": "Multiple techniques possible - requires analyst triage",
        "actions": [
            "Isolate the affected host or session while the event is investigated.",
            "Correlate with other alerts from the same source in the last 24 hours.",
            "Capture and preserve related network traffic for forensic review.",
            "Escalate to a senior analyst if the behaviour repeats.",
        ],
    },
    "Normal Activity": {
        "summary": "The model classified this event as consistent with normal behaviour. No malicious pattern was identified.",
        "mitre": "Not applicable",
        "actions": [
            "No action required.",
            "Keep the event in the audit log for baseline comparison.",
        ],
    },
}


# ---------------------------------------------------------------------------
# INFERENCE + EXPLAINABILITY
# ---------------------------------------------------------------------------
def to_frame(log):
    return pd.DataFrame([[log[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)


def predict_threat(log):
    """Return (class_id, confidence) for one event (dict of feature -> value)."""
    df = to_frame(log)
    class_id = int(model.predict(df)[0])
    confidence = float(model.predict_proba(df)[0][CLASS_INDEX[class_id]])
    return class_id, confidence


SEVERITY_BASE = {1: 2, 2: 1, 3: 2}   # index in RISK_ORDER: Brute Force=High, Port Scanning=Medium, Suspicious=High


def assess_risk(log, class_id, confidence):
    """
    Risk = how urgent an event is (the class says WHAT it is).
      1. start from the threat type's base severity
      2. +1 if its packet rate is in the top 25% of events of that type, -1 if in the lowest 25%
      3. -1 if the model is not confident (below 60%)
    Returns (risk level, plain-English rationale).
    """
    if class_id == 0:
        return "Low", "Classified as normal activity, so no threat risk is assigned."

    level = SEVERITY_BASE[class_id]
    notes = [f"{CLASS_NAMES[class_id]} has a base severity of {RISK_ORDER[level]}."]

    reference = RATE_REFERENCE.get(class_id)
    if reference is not None and len(reference):
        percentile = np.searchsorted(reference, log["Flow_Packets_per_s"], side="right") / len(reference)
        if percentile >= 0.75:
            level += 1
            notes.append("Its packet rate is in the top 25% for this threat type (+1).")
        elif percentile <= 0.25:
            level -= 1
            notes.append("Its packet rate is in the lowest 25% for this threat type (-1).")
    if confidence < 0.60:
        level -= 1
        notes.append(f"Model confidence is only {confidence * 100:.0f}% (-1).")

    level = max(0, min(level, len(RISK_ORDER) - 1))
    return RISK_ORDER[level], " ".join(notes)


def class_shap_values(input_df, class_id):
    """SHAP values of the predicted class (handles all SHAP output shapes)."""
    values = explainer.shap_values(input_df, check_additivity=False)
    idx = CLASS_INDEX[class_id]
    if isinstance(values, list):               # older SHAP: list of (n, features)
        return np.asarray(values[idx])[0]
    values = np.asarray(values)
    if values.ndim == 3:                       # newer SHAP: (n, features, classes)
        return values[0, :, idx]
    return values[0]


def build_explanation(log, shap_vals):
    """Return (plain-English reasons, ranked contributions for the chart)."""
    contributions = [
        {
            "feature": name,
            "label": FEATURE_LABELS[name],
            "value": log[name],
            "display": format_value(name, log[name]),
            "impact": round(float(shap_vals[i]), 4),
        }
        for i, name in enumerate(FEATURE_NAMES)
    ]
    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)

    drivers = sorted((c for c in contributions if c["impact"] >= MIN_REASON_IMPACT),
                     key=lambda c: c["impact"], reverse=True)[:MAX_REASONS]
    if not drivers:                            # fall back to the single strongest supporting feature
        drivers = sorted(contributions, key=lambda c: c["impact"], reverse=True)[:1]

    reasons = [describe_feature(c["feature"], c["value"], BASELINE[c["feature"]]["benign_median"])
               for c in drivers]
    return reasons, contributions


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------
@contextmanager
def get_db():
    """Open a connection that always commits/closes and returns dict-like rows."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def pick_replay_event(class_id=None):
    """Pick a held-out CICIDS2017 test event (never seen during training)."""
    available = [c for c, pool in DEMO_BY_CLASS.items() if len(pool)]
    if class_id not in available:
        class_id = random.choices(available, weights=[REPLAY_WEIGHTS.get(c, 0.25) for c in available])[0]
    row = DEMO_BY_CLASS[class_id].sample(1).iloc[0]
    log = {n: round(float(row[n]), 4) for n in FEATURE_NAMES}
    return log, int(class_id), str(row["raw_label"])


def insert_alert(conn, log, actual_class, actual_raw):
    """Classify an event with the model and store it as an alert."""
    class_id, confidence = predict_threat(log)
    risk, _ = assess_risk(log, class_id, confidence)
    ip = random_ip()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor = conn.execute(
        """INSERT INTO alerts (threat_type, ip_address, risk_level, timestamp, log_data, actual_label, actual_raw)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (CLASS_NAMES[class_id], ip, risk, timestamp, json.dumps(log), CLASS_NAMES[actual_class], actual_raw),
    )
    alert = {"id": cursor.lastrowid, "type": CLASS_NAMES[class_id], "ip": ip, "risk": risk, "time": timestamp,
             "correct": class_id == actual_class}
    return class_id, confidence, alert


def init_db():
    with get_db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS alerts (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   threat_type TEXT NOT NULL,
                   ip_address  TEXT NOT NULL,
                   risk_level  TEXT NOT NULL,
                   timestamp   TEXT NOT NULL,
                   log_data    TEXT NOT NULL,
                   actual_label TEXT,
                   actual_raw   TEXT)"""
        )
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(alerts)")}
        for col in ("actual_label", "actual_raw"):          # migrate databases from older versions
            if col not in existing:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {col} TEXT")

        # Alerts from the old 5-feature prototype stored a JSON list; the new model cannot explain them.
        conn.execute("DELETE FROM alerts WHERE log_data LIKE '[%'")

        # One-time upgrade (schema version 2): re-assess stored alerts with the new risk rules.
        if conn.execute("PRAGMA user_version").fetchone()[0] < 2:
            for r in conn.execute("SELECT id, log_data FROM alerts").fetchall():
                try:
                    log = json.loads(r["log_data"])
                except ValueError:
                    continue
                if not isinstance(log, dict) or any(n not in log for n in FEATURE_NAMES):
                    continue
                class_id, confidence = predict_threat(log)
                conn.execute("UPDATE alerts SET risk_level = ? WHERE id = ?",
                             (assess_risk(log, class_id, confidence)[0], r["id"]))
            conn.execute("PRAGMA user_version = 2")

        if conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0:
            for class_id in (1, 2, 3):                       # start with one event per attack class
                insert_alert(conn, *pick_replay_event(class_id))


init_db()


TIMELINE_MINUTES = 20


def get_stats():
    """Aggregate numbers used by the dashboard cards, charts and live updates."""
    now = datetime.now()
    minutes = [(now - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M") for i in range(TIMELINE_MINUTES - 1, -1, -1)]
    slot = {m: i for i, m in enumerate(minutes)}
    series = {level: [0] * TIMELINE_MINUTES for level in RISK_ORDER}

    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        by_risk = dict(conn.execute("SELECT risk_level, COUNT(*) FROM alerts GROUP BY risk_level").fetchall())
        by_threat = dict(conn.execute("SELECT threat_type, COUNT(*) FROM alerts GROUP BY threat_type").fetchall())
        checked, correct = conn.execute(
            """SELECT COUNT(*), COALESCE(SUM(CASE WHEN threat_type = actual_label THEN 1 ELSE 0 END), 0)
               FROM alerts WHERE actual_label IS NOT NULL""").fetchone()
        for minute, level, count in conn.execute(
                "SELECT substr(timestamp, 1, 16), risk_level, COUNT(*) FROM alerts WHERE timestamp >= ? GROUP BY 1, 2",
                (minutes[0],)):
            if minute in slot and level in series:
                series[level][slot[minute]] = count
        recent = [r[0] for r in conn.execute("SELECT risk_level FROM alerts ORDER BY id DESC LIMIT 10")]
        port_rows = conn.execute(
            "SELECT log_data FROM alerts WHERE threat_type != ? ORDER BY id DESC LIMIT 500", (CLASS_NAMES[0],)).fetchall()

    # Most targeted destination ports among detected threats
    port_counts = {}
    for r in port_rows:
        try:
            port = int(json.loads(r["log_data"])["Destination_Port"])
        except (ValueError, KeyError, TypeError):
            continue
        port_counts[port] = port_counts.get(port, 0) + 1
    top = sorted(port_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    peak = top[0][1] if top else 1
    top_ports = [{"port": p, "service": SERVICE_PORTS.get(p, ""), "count": c, "pct": round(100 * c / peak)}
                 for p, c in top]

    # Overall threat level from the 10 most recent alerts
    critical_recent = recent.count("Critical")
    high_recent = critical_recent + recent.count("High")
    if critical_recent >= 3 or high_recent >= 7:
        threat_level = {"label": "HIGH", "tone": "red"}
    elif critical_recent >= 1 or high_recent >= 3:
        threat_level = {"label": "ELEVATED", "tone": "amber"}
    else:
        threat_level = {"label": "NORMAL", "tone": "green"}

    return {
        "total": total,
        "critical": by_risk.get("Critical", 0) + by_risk.get("High", 0),
        "by_risk": {level: by_risk.get(level, 0) for level in RISK_ORDER},
        "by_threat": by_threat,
        "live_accuracy": round(100 * correct / checked, 1) if checked else None,
        "timeline": {"labels": [m[-5:] for m in minutes], "series": series},
        "top_ports": top_ports,
        "threat_level": threat_level,
    }


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            login_user(User(1))
            next_page = request.args.get("next")
            return redirect(next_page if is_safe_redirect(next_page) else url_for("dashboard"))
        flash("Invalid credentials.", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, threat_type, ip_address, risk_level, timestamp, actual_label FROM alerts ORDER BY id DESC"
        ).fetchall()

    alerts = [
        {"id": r["id"], "type": r["threat_type"], "ip": r["ip_address"], "risk": r["risk_level"],
         "time": r["timestamp"],
         "correct": None if r["actual_label"] is None else r["actual_label"] == r["threat_type"]}
        for r in rows
    ]
    return render_template("index.html", alerts=alerts, stats=get_stats(), model=MODEL_INFO, user=current_user)


@app.route("/api/stats")
@login_required
def api_stats():
    """JSON endpoint so charts can refresh without a page reload."""
    return jsonify(get_stats())


# GET is kept so a plain link still works; the dashboard calls it with POST.
@app.route("/simulate", methods=["GET", "POST"])
@login_required
def simulate():
    """Replay a real held-out CICIDS2017 event through the model."""
    log, actual_class, actual_raw = pick_replay_event()
    with get_db() as conn:
        class_id, confidence, alert = insert_alert(conn, log, actual_class, actual_raw)

    if request.method == "POST" and request.headers.get("X-Requested-With") == "fetch":
        return jsonify({
            "status": "ok",
            "threat": CLASS_NAMES[class_id],
            "confidence": round(confidence, 3),
            "actual": actual_raw,
            "correct": class_id == actual_class,
            "alert": alert,
        })
    return redirect(url_for("dashboard"))


@app.route("/alert/<int:alert_id>")
@login_required
def alert_detail(alert_id):
    with get_db() as conn:
        row = conn.execute(
            """SELECT threat_type, ip_address, risk_level, timestamp, log_data, actual_label, actual_raw
               FROM alerts WHERE id = ?""", (alert_id,)).fetchone()

    if row is None:
        abort(404)

    log = json.loads(row["log_data"])
    if not isinstance(log, dict) or any(n not in log for n in FEATURE_NAMES):
        flash("This alert was recorded with an older model version and cannot be explained.", "danger")
        return redirect(url_for("dashboard"))

    class_id, confidence = predict_threat(log)
    _, risk_reason = assess_risk(log, class_id, confidence)
    shap_vals = class_shap_values(to_frame(log), class_id)
    reasons, contributions = build_explanation(log, shap_vals)

    alert = {
        "id": alert_id,
        "type": row["threat_type"],
        "ip": row["ip_address"],
        "risk": row["risk_level"],
        "time": row["timestamp"],
        "confidence": round(confidence * 100, 1),
        "actual": row["actual_label"],
        "actual_raw": row["actual_raw"],
        "correct": None if row["actual_label"] is None else row["actual_label"] == row["threat_type"],
    }
    return render_template(
        "alert_detail.html",
        alert=alert,
        reasons=reasons,
        contributions=contributions,  # ranked SHAP values for the bar chart
        playbook=THREAT_PLAYBOOK.get(row["threat_type"]),
        risk_reason=risk_reason,
    )


SHORT_NAMES = {"Normal Activity": "Normal", "Brute Force Attack": "Brute Force",
               "Port Scanning": "Port Scan", "Suspicious Attack": "Suspicious"}


@app.route("/performance")
@login_required
def performance():
    """Model evaluation page: metrics, confusion matrix and global SHAP importance."""
    m = metrics
    n_test = m["n_test"]

    rows = []
    for name in m["class_names"]:
        r = m["report"][name]
        rows.append({"name": name, "precision": r["precision"], "recall": r["recall"],
                     "f1": r["f1-score"], "support": int(r["support"]),
                     "share": 100 * r["support"] / n_test})

    matrix = []
    for i, counts in enumerate(m["confusion_matrix"]):
        total = max(sum(counts), 1)
        cells = []
        for j, count in enumerate(counts):
            frac = count / total
            if i == j:
                bg = f"rgba(34,211,238,{0.12 + 0.7 * frac:.2f})"
            else:
                bg = "transparent" if count == 0 else f"rgba(239,68,68,{0.12 + min(0.7, frac * 6):.2f})"
            cells.append({"count": count, "pct": 100 * frac, "diag": i == j, "bg": bg})
        matrix.append({"name": m["class_names"][i], "cells": cells})

    gini = m["feature_importance"]
    shap_imp = m.get("global_shap") or {}
    order = sorted(FEATURE_NAMES, key=(lambda n: shap_imp[n]) if shap_imp else (lambda n: gini[n]), reverse=True)
    shap_total = sum(shap_imp.values()) or 1
    chart = {
        "labels": [FEATURE_LABELS[n] for n in order],
        "gini": [round(100 * gini[n], 1) for n in order],
        "shap": [round(100 * shap_imp[n] / shap_total, 1) for n in order] if shap_imp else [],
    }

    # Data-driven key findings (these change automatically when the model is retrained)
    accuracy, baseline = m["accuracy"] * 100, m["majority_baseline"] * 100
    majority_name = max(m["test_distribution"], key=m["test_distribution"].get)
    insights = [
        f"Overall accuracy is {accuracy:.2f}%. A model that always predicted '{majority_name}' would already score "
        f"{baseline:.2f}%, so the real improvement is {accuracy - baseline:.2f} percentage points. "
        f"Macro F1 ({MODEL_INFO['macro_f1']}) weighs every class equally and is the fairer summary."
    ]
    weakest = min(rows, key=lambda r: r["f1"])
    if weakest["f1"] < 0.90:
        if weakest["recall"] >= weakest["precision"]:
            text = (f"{weakest['name']} is the weakest class (F1 {weakest['f1']:.2f}). The model catches most real cases "
                    f"(recall {weakest['recall']:.2f}) but also raises false alarms (precision {weakest['precision']:.2f}).")
        else:
            text = (f"{weakest['name']} is the weakest class (F1 {weakest['f1']:.2f}). Its alerts are usually right "
                    f"(precision {weakest['precision']:.2f}) but some real cases are missed (recall {weakest['recall']:.2f}).")
        if weakest["share"] < 5:
            text += f" The class is rare ({weakest['share']:.1f}% of the test set, {weakest['support']:,} flows), which makes high precision harder."
        insights.append(text)
    else:
        insights.append(f"Every class scores an F1 above 0.90; the weakest is {weakest['name']} (F1 {weakest['f1']:.2f}).")
    if shap_imp:
        top = sorted(shap_imp, key=shap_imp.get, reverse=True)[:2]
        insights.append(
            f"{FEATURE_LABELS[top[0]]} has the largest overall influence on decisions "
            f"({100 * shap_imp[top[0]] / shap_total:.0f}% of total SHAP impact), followed by "
            f"{FEATURE_LABELS[top[1]]} ({100 * shap_imp[top[1]] / shap_total:.0f}%).")
    insights.append("These scores come from a random stratified split of a single dataset. Performance on other networks "
                    "or newer attack types may be lower, so they should be read as benchmark results, not guarantees.")

    params = model.get_params()
    train = [
        ("Algorithm", "Random Forest"),
        ("Trees", params["n_estimators"]),
        ("Max depth", params["max_depth"]),
        ("Min samples per leaf", params["min_samples_leaf"]),
        ("Class weighting", params["class_weight"]),
        ("Split", f"80% train ({m['n_train']:,}) / 20% test ({n_test:,}), stratified"),
        ("Training time", f"{m['training_seconds']} s"),
        ("Trained on", m["trained_at"]),
    ]
    return render_template("performance.html", m=m, info=MODEL_INFO, rows=rows, matrix=matrix,
                           short=SHORT_NAMES, chart=chart, insights=insights, train=train,
                           baseline_name=majority_name, accuracy=accuracy, baseline=baseline)


@app.route("/about")
@login_required
def about():
    """System overview / methodology page."""
    return render_template("about.html", model=MODEL_INFO, playbook=THREAT_PLAYBOOK, stats=get_stats())


@app.errorhandler(404)
def not_found(_error):
    return "Page not found", 404


if __name__ == "__main__":
    app.run(debug=DEBUG)