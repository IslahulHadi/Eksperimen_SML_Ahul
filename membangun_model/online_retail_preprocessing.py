# -*- coding: utf-8 -*-
"""online_retail_preprocessing.py

Proyek Customer Segmentation oleh Ahul (IslahulHadi)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    # 1. Load Dataset
    df = pd.read_csv(file_path)

    # 2. Cleaning (Contoh: Menghapus nilai kosong)
    df = df.dropna()

    # 3. Feature Selection (Pilih kolom yang Anda gunakan)
    # Sesuaikan dengan kolom yang Anda pakai di modelling_tuning.py
    features = df[['Quantity', 'UnitPrice']]

    # 4. Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    # Simpan hasil preprocessing ke CSV baru
    pd.DataFrame(scaled_data).to_csv('OnlineRetail_preprocessed.csv', index=False)
    print("Preprocessing Selesai!")

if __name__ == "__main__":
    preprocess_data('Online_Retail.csv') # Sesuaikan nama file asli Anda