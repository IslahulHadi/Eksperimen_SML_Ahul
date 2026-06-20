# -*- coding: utf-8 -*-
"""inference.py

Proyek Customer Segmentation oleh Ahul (IslahulHadi)
"""

import mlflow
import pandas as pd

# Pastikan Anda sudah login ke DagsHub di terminal/Colab sebelumnya
# Ganti URL di bawah dengan 'Model URI' yang ada di tab Artifacts DagsHub Anda
model_uri = "models:/retail_segmentation_model/latest"
model = mlflow.pyfunc.load_model(model_uri)

# Membuat data contoh (Quantity, UnitPrice)
data_baru = pd.DataFrame([[10, 2.5]], columns=['Quantity', 'UnitPrice'])

# Melakukan prediksi
prediksi = model.predict(data_baru)
print(f"Hasil Segmentasi Cluster: {prediksi}")