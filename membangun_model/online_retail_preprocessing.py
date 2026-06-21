# -*- coding: utf-8 -*-
"""online_retail_preprocessing.py

Proyek Customer Segmentation oleh Ahul (IslahulHadi)

Script ini melakukan preprocessing lengkap dari dataset mentah:
1. Load Dataset
2. Cleaning (missing values, duplikat, data invalid)
3. Feature Engineering (TotalAmount)
4. Membangun RFM (Recency, Frequency, Monetary)
5. Menyimpan hasil RFM ke CSV
"""

import pandas as pd
import numpy as np


def preprocess_data(file_path):
    """
    Preprocessing dataset Online Retail hingga menghasilkan tabel RFM.

    Parameters:
        file_path (str): Path ke file OnlineRetail.csv (dataset mentah)

    Returns:
        DataFrame: Tabel RFM per CustomerID
    """
    print("=" * 60)
    print("PREPROCESSING — Customer Segmentation")
    print("=" * 60)

    # ============================================================
    # 1. LOAD DATASET
    # ============================================================
    print("\n[STEP 1] Load Dataset...")
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"  Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")

    # ============================================================
    # 2. CLEANING
    # ============================================================
    print("\n[STEP 2] Cleaning Data...")

    # Drop missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    print(f"  Setelah drop missing CustomerID: {len(df)} baris")

    # Drop duplicates
    df = df.drop_duplicates()
    print(f"  Setelah drop duplikat: {len(df)} baris")

    # Hapus transaksi batal (InvoiceNo dimulai 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Hapus Quantity <= 0 dan UnitPrice <= 0
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    print(f"  Setelah hapus data invalid: {len(df)} baris")

    # ============================================================
    # 3. FEATURE ENGINEERING
    # ============================================================
    print("\n[STEP 3] Feature Engineering...")

    # Konversi InvoiceDate ke datetime
    df['InvoiceDate'] = pd.to_datetime(
        df['InvoiceDate'],
        format='mixed',
        dayfirst=True,
        errors='coerce'
    )

    # Buat kolom TotalAmount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    # ============================================================
    # 4. BANGUN RFM
    # ============================================================
    print("\n[STEP 4] Membangun RFM...")

    # Tanggal referensi (hari terakhir transaksi)
    ref_date = df['InvoiceDate'].max()

    # Tabel RFM per CustomerID
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,   # Recency
        'InvoiceNo': 'nunique',                                # Frequency
        'TotalAmount': 'sum'                                   # Monetary
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    print(f"  Jumlah customer: {rfm.shape[0]}")
    print(f"\n  Statistik RFM:")
    print(rfm.describe())

    # ============================================================
    # 5. SIMPAN HASIL
    # ============================================================
    print("\n[STEP 5] Simpan Hasil...")

    # Simpan data preprocessed
    df.to_csv('OnlineRetail_preprocessed.csv', index=False)
    print("  Data preprocessed tersimpan: OnlineRetail_preprocessed.csv")

    # Simpan RFM
    rfm.to_csv('OnlineRetail_RFM.csv')
    print("  Data RFM tersimpan: OnlineRetail_RFM.csv")

    print("\n" + "=" * 60)
    print("✅ Preprocessing Selesai!")
    print("=" * 60)

    return rfm


if __name__ == "__main__":
    preprocess_data('OnlineRetail.csv')