# spoilage_predictor.py
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from joblib import load
import pandas as pd
import numpy as np
from pathlib import Path

class SpoilagePredictor:
    def __init__(self,
                 scaler_path="models/spoilage_scaler.joblib",
                 iso_path="models/spoilage_iso_forest.joblib"):
        self.scaler = load(scaler_path)
        self.iso_forest = load(iso_path)

    def predict(self, temp_c: float, humidity_pct: float, delay_hr: int):
        row = pd.DataFrame([[temp_c, humidity_pct, delay_hr]],
                           columns=['temperature_C', 'humidity_%', 'transport_delay_hr'])

        X_scaled = self.scaler.transform(row)
        is_anomaly = self.iso_forest.predict(X_scaled)[0] == -1

        # simple rule-based probability
        prob = (
            0.04 +
            0.022 * max(0, temp_c - 4) +
            0.012 * max(0, humidity_pct - 70) +
            0.035 * (delay_hr / 12)
        )
        prob = min(max(prob, 0.0), 1.0)

        if prob >= 0.20 or is_anomaly:
            status = "CRITICAL"
        elif prob >= 0.10:
            status = "WARNING"
        else:
            status = "SAFE"

        return {
            "probability": round(prob, 4),
            "status": status,
            "is_anomaly": is_anomaly,
            "raw_values": {"temp": temp_c, "humidity": humidity_pct, "delay": delay_hr}
        }
