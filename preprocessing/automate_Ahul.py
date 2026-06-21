# -*- coding: utf-8 -*-
"""automate_Ahul.py

Script Automasi Preprocessing Dataset Online Retail
Proyek Customer Segmentation oleh Ahul (IslahulHadi)

Script ini melakukan:
1. Load dataset mentah (OnlineRetail.csv)
2. Cleaning (hapus missing, duplikat, data invalid)
3. Feature Engineering (TotalAmount, RFM)
4. Simpan hasil preprocessing
"""

import pandas as pd
import numpy as np
import os


def run_preprocessing(input_path, output_dir='.'):
    """
    Jalankan preprocessing dari dataset mentah hingga RFM.

    Parameters:
        input_path (str): Path ke file OnlineRetail.csv
        output_dir (str): Direktori output untuk menyimpan hasil
    """
    print("=" * 60)
    print("AUTOMASI PREPROCESSING — Customer Segmentation")
    print("=" * 60)

    # ============================================================
    # 1. LOAD DATASET
    # ============================================================
    print("\n[STEP 1] Load Dataset...")
    df = pd.read_csv(input_path, encoding='ISO-8859-1')
    print(f"  Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")

    # ============================================================
    # 2. CLEANING
    # ============================================================
    print("\n[STEP 2] Cleaning Data...")

    # Drop missing CustomerID
    df_clean = df.dropna(subset=['CustomerID'])
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    print(f"  Setelah drop missing CustomerID: {len(df_clean)} baris")

    # Drop duplicates
    df_clean = df_clean.drop_duplicates()
    print(f"  Setelah drop duplikat: {len(df_clean)} baris")

    # Hapus transaksi yang dibatalkan (InvoiceNo dimulai 'C')
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
    print(f"  Setelah hapus transaksi batal: {len(df_clean)} baris")

    # Hapus Quantity <= 0 dan UnitPrice <= 0
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    print(f"  Setelah hapus Quantity/UnitPrice <= 0: {len(df_clean)} baris")

    # ============================================================
    # 3. FEATURE ENGINEERING
    # ============================================================
    print("\n[STEP 3] Feature Engineering...")

    # Konversi InvoiceDate ke datetime
    df_clean['InvoiceDate'] = pd.to_datetime(
        df_clean['InvoiceDate'],
        format='mixed',
        dayfirst=True,
        errors='coerce'
    )

    # Buat TotalAmount
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

    # Simpan data yang sudah bersih
    cleaned_path = os.path.join(output_dir, 'OnlineRetail_preprocessed.csv')
    df_clean.to_csv(cleaned_path, index=False)
    print(f"  Data bersih tersimpan: {cleaned_path}")

    # ============================================================
    # 4. BANGUN RFM
    # ============================================================
    print("\n[STEP 4] Membangun RFM...")

    # Tanggal referensi
    ref_date = df_clean['InvoiceDate'].max()

    # Tabel RFM per CustomerID
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,   # Recency
        'InvoiceNo': 'nunique',                                # Frequency
        'TotalAmount': 'sum'                                   # Monetary
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    print(f"  Jumlah customer: {rfm.shape[0]}")
    print(f"\n  Statistik RFM:")
    print(rfm.describe())

    # Simpan RFM
    rfm_path = os.path.join(output_dir, 'OnlineRetail_RFM.csv')
    rfm.to_csv(rfm_path)
    print(f"\n  RFM tersimpan: {rfm_path}")

    print("\n" + "=" * 60)
    print("✅ PREPROCESSING SELESAI!")
    print("=" * 60)

    return rfm


if __name__ == "__main__":
    run_preprocessing('OnlineRetail.csv')
