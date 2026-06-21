# -*- coding: utf-8 -*-
"""inference.py

Script Inference untuk Model Customer Segmentation
Proyek oleh Ahul (IslahulHadi)
"""

import requests
import json

# URL endpoint Flask wrapper (prometheus_exporter.py)
PREDICT_URL = "http://127.0.0.1:8000/predict"

# Contoh data input RFM (Recency, Frequency, Monetary)
sample_input = {
    "instances": [
        [10, 50, 10000.0],    # Customer aktif, sering belanja, VIP
        [200, 2, 300.0],      # Customer tidak aktif, jarang belanja
        [30, 15, 5000.0],     # Customer loyal menengah
    ]
}


def predict(input_data):
    """Kirim request prediksi ke Flask wrapper."""
    headers = {"Content-Type": "application/json"}
    response = requests.post(PREDICT_URL, headers=headers, data=json.dumps(input_data))

    if response.status_code == 200:
        print("✅ Prediction:", response.json())
    else:
        print("❌ Error:", response.status_code, response.text)


if __name__ == "__main__":
    predict(sample_input)
