import json
import random
from datetime import datetime, timedelta

# Set seed so we get the same 100 alerts every time we run it
random.seed(42)

services = ["checkout-api", "auth-service", "billing-worker", "frontend", "database"]
severities = ["P1", "P2", "P3"]

alerts = []

# --- 1. Generate 80 Noise Alerts ---
for i in range(80):
    service = random.choice(services)
    alerts.append({
        "id": f"alert-noise-{i}",
        "service": service,
        "severity": random.choice(severities),
        "error_rate": round(random.uniform(0.01, 0.04), 3),  # Low error rate
        "error_rate_slope": round(random.uniform(-0.005, 0.002), 3), # Flat or dropping
        "recent_deploy_mins_ago": random.choice([None, 1440, 2880]), # Old or no deploy
        "open_incident": random.choice([True, False]), # Often duplicates
        "same_alert_count_24h": random.randint(2, 10), # Fired a lot
        "ground_truth": "noise"
    })

# --- 2. Generate 20 Real Incident Alerts ---
for i in range(20):
    service = random.choice(services)
    alerts.append({
        "id": f"alert-real-{i}",
        "service": service,
        "severity": random.choice(["P1", "P1", "P2"]), # Mostly high severity
        "error_rate": round(random.uniform(0.06, 0.15), 3), # High error rate
        "error_rate_slope": round(random.uniform(0.005, 0.02), 3), # Climbing fast
        "recent_deploy_mins_ago": random.choice([15, 30, 45, None]), # Recent deploy likely
        "open_incident": False, # New incident
        "same_alert_count_24h": 0, # First time seeing this
        "ground_truth": "real_incident"
    })

# Shuffle them so they are mixed
random.shuffle(alerts)

# Write to file
with open("data/alerts.jsonl", "w") as f:
    for alert in alerts:
        f.write(json.dumps(alert) + "\n")

print(f"Successfully generated {len(alerts)} alerts in data/alerts.jsonl")
print("80 noise alerts, 20 real incidents.")