# -*- coding: utf-8 -*-
"""Eksperimen_Ahul.ipynb

Proyek Customer Segmentation oleh Ahul (IslahulHadi)
"""

# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

"""# **LOAD DATA**"""

# load data
df = pd.read_csv('OnlineRetail.csv', encoding='latin1')

print("5 baris Pertama")
print(df.head())

print("\nINFO DATA:")
print(df.info())

print("\nDESKRIPSI STATISTIK:")
print(df.describe())

# Missing Values & Duplikasi
print("\nMissing Value:")
print(df.isnull().sum())

print("\nJumlah Duplikasi:", df.duplicated().sum())

#  Cek Nilai Unik di Kolom Penting

print("JUMLAH Nilai Unik: ")
print("Jumlah Invoice unik (transaksi berbeda):", df['InvoiceNo'].nunique())
print("Jumlah Customer unik (jumlah customer):", df['CustomerID'].nunique())
print("Jumlah Produk unik (variasi produk):", df['Description'].nunique())
print("Jumlah Kode Produk unik (variasi StockCode):", df['StockCode'].nunique())
print("Jumlah Negara unik:", df['Country'].nunique())

#  Datetime Analysis
df['InvoiceDate'] = pd.to_datetime(
    df['InvoiceDate'],
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

print("\nRange Waktu Transaksi:")
print("Transaksi Pertama:", df['InvoiceDate'].min())
print("Transaksi Terakhir:", df['InvoiceDate'].max())

#  waktu
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Hour'] = df['InvoiceDate'].dt.hour
df['Date'] = df['InvoiceDate'].dt.date

#  Analisis transaksi paling ramai
trans_per_hari = df.groupby('Date')['InvoiceNo'].count()
trans_per_bulan = df.groupby('Month')['InvoiceNo'].count()
trans_per_jam = df.groupby('Hour')['InvoiceNo'].count()

print("\nTanggal paling ramai:", trans_per_hari.idxmax(), "dengan", trans_per_hari.max(), "transaksi")
print("Bulan paling ramai:", trans_per_bulan.idxmax(), "dengan", trans_per_bulan.max(), "transaksi")
print("Jam paling ramai:", trans_per_jam.idxmax(), "dengan", trans_per_jam.max(), "transaksi")

#  Perhitungan Pendapatan (Revenue)
df['Revenue'] = df['Quantity'] * df['UnitPrice']

df['Revenue'] = df['Revenue'].round(2)

#  Hitung total revenue
total_revenue = df['Revenue'].sum()

print("\nTotal Revenue:", f"{total_revenue:,.2f}")

# Negara Terbanyak Transaksi
top_country = df['Country'].value_counts().head(10)

plt.figure(figsize=(8,4))
sns.barplot(x=top_country.values, y=top_country.index, palette="viridis")
plt.title("Top 10 Negara dengan Transaksi Terbanyak")
plt.xlabel("Jumlah Transaksi")
plt.ylabel("Negara")
plt.show()

# Top Product berdasarkan revenue
top_products = df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_products.values, y=top_products.index, palette="magma")
plt.title("Top 10 Produk Berdasarkan Revenue")
plt.xlabel("Revenue")
plt.ylabel("Produk")
plt.show()

#  Analisis Customer
#  Customer paling aktif
active_customer = df.groupby('CustomerID')['InvoiceNo'].nunique().sort_values(ascending=False).head(10)
print("\nTop 10 Customer Paling Aktif:")
print(active_customer)

#  Customer dengan pembelian terbanyak
top_cust_qty = df.groupby('CustomerID')['Quantity'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Customer dengan Quantity Terbanyak:")
print(top_cust_qty)

plt.figure(figsize=(8,4))
sns.boxplot(x=df['Quantity'], color='skyblue')
plt.title("Distribusi Quantity")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x=df['UnitPrice'], color='salmon')
plt.title("Distribusi UnitPrice")
plt.show()

"""# **PREPROCESSING**

**Drop Missing Value**
"""

df_drop = df.dropna(subset=['CustomerID'])
df_drop['CustomerID'] = df_drop['CustomerID'].astype(int)

"""**Drop Duplicate**"""

df_duplicate = df_drop.drop_duplicates()

print('Missing value: ', df_duplicate.isnull().sum())
print('Duplikasi: ', df_duplicate.duplicated().sum())

"""**Bersihkan Data Invalid**"""

df_step1 = df_duplicate[~df_duplicate['InvoiceNo'].astype(str).str.startswith('C')]

#  Remove quantity <= 0
df_clean = df_step1[df_step1['Quantity'] > 0]

#  Remove UnitPrice <= 0
df_clean = df_clean[df_clean['UnitPrice'] > 0]

#  Remove extreme outlier
Q1_Q = df_clean['Quantity'].quantile(0.25)
Q3_Q = df_clean['Quantity'].quantile(0.75)
IQR_Q = Q3_Q - Q1_Q

lower_Q = Q1_Q - 1.5 * IQR_Q
upper_Q = Q3_Q + 1.5 * IQR_Q

Q1_P = df_clean['UnitPrice'].quantile(0.25)
Q3_P = df_clean['UnitPrice'].quantile(0.75)
IQR_P = Q3_P - Q1_P

lower_P = Q1_P - 1.0 * IQR_P
upper_P = Q3_P + 1.0 * IQR_P

df_final = df_clean[
    (df_clean['Quantity'] >= lower_Q) & (df_clean['Quantity'] <= upper_Q) &
    (df_clean['UnitPrice'] >= lower_P) & (df_clean['UnitPrice'] <= upper_P)
]

df_final[['Quantity', 'UnitPrice']].describe()

plt.figure(figsize=(10,5))

plt.subplot(121)
sns.boxplot(df_final['Quantity'])
plt.title("Distribusi Quantity Setelah Cleaning")

plt.subplot(122)
sns.boxplot(df_final['UnitPrice'])
plt.title("Distribusi UnitPrice Setelah Cleaning")

plt.tight_layout()
plt.show()

"""**Konversi InvoiceDate menjadi tipe datetime**

Agar bisa menghitung Recency di RFM dan mengurutkan data berdasarkan waktu


"""

df_final1 = df_final.copy()
df_final1['InvoiceDate'] = pd.to_datetime(df_final1['InvoiceDate'])

df_final1['InvoiceDate'].dtype

"""**Membuat Fitur TotalAmount**"""

df_final2 = df_final1.copy()
df_final2['TotalAmount'] = df_final2['Quantity'] * df_final2['UnitPrice']

df_final2[['Quantity', 'UnitPrice', 'TotalAmount']].head()

df_final2['TotalAmount'].describe()

print("Jumlah customer sebelum cleaning:", df['CustomerID'].nunique())
print("Jumlah customer setelah cleaning:", df_final2['CustomerID'].nunique())

"""**Membangun RFM (Recency, Frequency, Monetary)**"""

#  Tanggal referensi
ref_date = df_final2['InvoiceDate'].max()
ref_date

#  Tabel RFM
rfm = df_final2.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days,    # Recency
    'InvoiceNo': 'nunique',                                # Frequency
    'TotalAmount': 'sum'                                   # Monetary
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm.head()

rfm.describe()

#  rfm_log = np.log1p(rfm)

"""**Normalisasi RFM**"""

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

rfm_scaled_df = pd.DataFrame(
    rfm_scaled,
    index = rfm.index,
    columns = ['Recency', 'Frequency', 'Monetary']
)

rfm_scaled_df.describe()

sns.boxplot(rfm['Monetary'])

"""# **Elbow Method**"""

# Range Jumlah Cluster (K)
k_range = range(2,11)

inertia_list = []

for k in k_range:
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(rfm_scaled_df)
  inertia_list.append(kmeans.inertia_)

# Plot Elbow
plt.figure(figsize=(8,5))
plt.plot(k_range, inertia_list, marker='o', linestyle='--', color='red')
plt.title('Elbow Method')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia / SSE')
plt.xticks(k_range)
plt.grid(True)
plt.show()

"""# **Silhouette Score**"""

X = rfm_scaled_df

silhouette_scores = []
K = range(2, 11)

print("Silhouette Score untuk setiap jumlah cluster:")
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K = {k}: Silhouette Score = {score:.4f}")

# Plot Grafik Silhouete Score
plt.figure(figsize=(8,5))
plt.plot(K, silhouette_scores, marker='o', linestyle='--', color='red')
plt.title('Silhouette Score VS Jumlah Cluster (K)')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

"""# **CLUSTERING (NYOBA)**"""

# k=4 karena dengan nilai sihoulette tertinggi
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled_df)

rfm

rfm.groupby('Cluster').mean()

"""**Cluster 2 — High-Value Customers / VIP**

 **Karakteristik:**
- **Recency sangat rendah (2,5 hari)** → baru saja bertransaksi  
- **Frequency sangat tinggi (147)** → Pelanggan sangat sering berbelanja, jauh di atas cluster lainnya.
- **Monetary sangat tinggi (42.091)** → nilai pembelian terbesar di semua cluster

Kategori:
**High-Value Customers**

**Pelanggan Juara (Champions)**
Cluster ini mewakili inti dari kesuksesan Glossier Skincare. Mereka adalah pelanggan terbaik dan paling bernilai (RFM Score Tertinggi).

**Insight:**

Cluster ini merupakan pelanggan terbaik dan paling bernilai bagi bisnis. Mereka tidak hanya aktif (Recency Rendah), tetapi juga memiliki loyalitas tinggi (Frequency Tinggi) dan daya beli sangat besar (Monetary Tinggi). Kehilangan pelanggan dalam cluster ini akan berdampak besar terhadap pendapatan perusahaan.

**Strategi Bisnis:**

**1. Premium Membership:** Segera upgrade ke Program Loyalitas Gold/Diamond Glossier Skincare dengan benefit eksklusif.

**2. Layanan Prioritas:** Prioritaskan mereka dalam layanan pelanggan, pengiriman ekspres, dan undangan preview produk baru (VIP Access).

**3. Penawaran Eksklusif:** Kirimkan personalized gift atau penawaran produk edisi terbatas (tidak diskon) yang memvalidasi status mereka.

**Tujuan:**

Mempertahankan retention >95% dan mendorong advocacy.

---

**Cluster 3 — Loyal Mid-Value Customers / Potential High-Value**

 **Karakteristik:**
- **Recency rendah (21 Hari)** → pelanggan aktif  
- **Frequency cukup tinggi (18,43)**  
- **Monetary tinggi (5.034)** →
Nilai belanja relatif besar, meskipun masih di bawah cluster VIP

 Kategori:
**Loyal Customers / Potential High-Value**

**Pelanggan Setia**
Mereka adalah pelanggan reguler yang menyukai merek Anda tetapi belum mencapai tingkat spending Pelanggan Juara. Mereka memiliki potensi up-selling yang sangat tinggi.

**Insight:**

Cluster ini memiliki Frequency Tinggi dan Recency yang baik, tetapi nilai Monetary masih bisa ditingkatkan. Mereka adalah pengguna rutin yang puas, dan fokus strategi harus diarahkan pada peningkatan nilai transaksi rata-rata (AOV).

**Strategi Bisnis:**

**1. Rekomendasi Rutinitas Step-up:** Jika mereka rutin membeli cleanser, rekomendasikan serum atau toner pelengkap (produk bernilai lebih tinggi) melalui personalized email.

**2. Smart Bundling:** Tawarkan bundling produk yang saling melengkapi (misalnya, paket night routine) dengan sedikit insentif, bukan diskon besar.

**3. Uji Coba Produk Baru:** Beri mereka sample size dari produk premium yang baru diluncurkan.

**Tujuan:**

Meningkatkan Monetary mereka sebesar 10% dan mendorong up-selling produk bernilai tinggi.

---

** Cluster 0 — Active Low-Spenders**

 **Karakteristik:**
- **Recency menengah (51,27 hari)** → masih aktif  
- **Frequency rendah (3.46)**  
- **Monetary menengah (820)**  

 Kategori:
**Active Low-Spenders / Regular Customers**

**Pembeli Frekuentif (Frequent Buyers)**
Segmen ini aktif dan sering membeli, tetapi nilai setiap transaksi mereka cenderung kecil. Mereka sangat responsif terhadap promosi.

**Insight:**

Cluster ini sering berinteraksi dengan merek (Frequency Tinggi) dan Recency mereka bagus, tetapi nilai belanjanya (Monetary) rendah. Mereka cenderung membeli produk low-cost atau saat ada diskon kecil. Mereka price-sensitive.

**Strategi Bisnis:**

**1. Edukasi Nilai:** Alih-alih diskon, fokus pada edukasi tentang manfaat produk premium (misalnya, webinar singkat tentang skincare investment).

**2. Sample Kits Produk Premium:** Kirimkan sample kits dari produk mahal sebagai bagian dari pembelian berikutnya. Tujuannya adalah membuat mereka mencoba, menyukai, dan membeli produk bernilai tinggi di masa depan.

**3. Target Threshold:** Tawarkan insentif kecil (misalnya, free shipping) jika mereka mencapai nilai transaksi tertentu yang sedikit lebih tinggi dari AOV mereka saat ini.

**Tujuan:**

Mengonversi frekuensi menjadi nilai, meningkatkan Monetary jangka panjang.

---
**Cluster 1 — Inactive / Low-Value Customers**

 **Karakteristik:**
- **Recency sangat tinggi (261,75 hari)** → sudah lama tidak berbelanja  
- **Frequency sangat rendah (1.50)**  
- **Monetary rendah (313)**  

 Kategori:
**Inactive Customers / At-Risk / Lost Customers**

**Pelanggan Berisiko**
Ini adalah segmen krusial yang perlu diintervensi segera untuk mencegah churn permanen.

**Insight:**

Cluster ini memiliki Recency Tinggi (Lama tidak berbelanja) dan Monetary yang bervariasi. Mereka adalah pelanggan yang nilainya sedang hingga tinggi di masa lalu, tetapi kini menunjukkan tanda-tanda menghilang. Kehilangan mereka adalah kerugian yang dapat dicegah.

**Strategi Bisnis:**

Voucher Reaktivasi Kuat: Kirimkan Voucher Diskon Besar (misalnya, 30%) yang sangat kuat.

**1. Unsur Urgensi:** Tambahkan Batas Waktu yang ketat (expired in 7 days) dan narasi personal, seperti: "Kami Merindukan Rutinitas Kulit Anda, Gunakan Voucher Ini Sekarang!"

**2. Survei Reaktivasi:** Jika mereka tidak merespons voucher, kirimkan survei singkat untuk memahami alasan churn (misalnya, pindah merek, masalah kualitas, harga).

**Tujuan:**

Reaktivasi segera (target 15%) dan mencegah churn menjadi permanen.

"""



"""**Visualisasi Cluster**"""

plt.figure(figsize=(16,8))

plt.subplot(131)
sns.boxplot(x='Cluster', y='Recency', data=rfm)
plt.title("Recency per Cluster")

plt.subplot(132)
sns.boxplot(x='Cluster', y='Frequency', data=rfm)
plt.title("Frequency per Cluster")

plt.subplot(133)
sns.boxplot(x='Cluster', y='Monetary', data=rfm)
plt.title("Monetary per Cluster")

plt.tight_layout()
plt.show()

"""**1. Outlier pada Monetary**
- Muncul karena beberapa pelanggan melakukan pembelian dalam jumlah besar.
- Mewakili pelanggan high-value seperti reseller atau pembeli grosir.
- Informasi penting karena menunjukkan kontribusi pelanggan prioritas.

**2. Outlier pada Frequency**
- Disebabkan oleh pelanggan yang sering melakukan transaksi berulang.
- Menggambarkan kelompok pelanggan loyal.
- Tidak boleh dihapus karena berpengaruh pada segmentasi pelanggan.

**3. Outlier pada Recency**
- Nilai Recency tinggi menunjukkan pelanggan tidak aktif dalam waktu lama.
- Penting untuk mengidentifikasi pelanggan yang perlu strategi re-engagement.
"""

plt.figure(figsize=(16,8))

plt.subplot(131)
sns.scatterplot(x='Recency', y='Frequency', hue='Cluster', data=rfm)
plt.title("Recency vs Frequency")

plt.subplot(132)
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm)
plt.title("Recency vs Monetary")

plt.subplot(133)
sns.scatterplot(x='Frequency', y='Monetary', hue='Cluster', data=rfm)
plt.title("Frequency vs Monetary")

plt.tight_layout()
plt.show()

"""**  1. Scatter Plot: Recency vs Frequency**

Plot ini menunjukkan hubungan antara:
- Recency → waktu sejak terakhir transaksi
- Frequency → seberapa sering pelanggan bertransaksi


- **Cluster 2 (High-Value)**  
  - Frequency sangat tinggi (hingga >100).  
  - Recency sangat rendah.  
  - Pelanggan sangat aktif dan bertransaksi dalam jumlah besar.

- **Cluster 3 (Loyal Medium-High Value Customers)**  
  - Frequency menengah (10–50).  
  - Recency rendah.  
  - Pelanggan aktif dan loyal dengan kontribusi stabil.

- **Cluster 1 (Inactive / Lost Customers)**  
  - Frequency sangat rendah dan Recency sangat tinggi.  
  - Pelanggan sudah lama tidak kembali bertransaksi.  

- **Cluster 0 (Active Low-Spenders)**  
  - Recency rendah–menengah namun Frequency rendah.  
  - Pelanggan masih aktif namun nilai belanjanya kecil.

---

**  2. Scatter Plot: Recency vs Monetary**

Plot ini menunjukkan hubungan antara:
- Recency
- Monetary → total nilai transaksi pelanggan


- **Cluster 2 (High-Value)**  
  - Monetary sangat tinggi, termasuk outlier >80.000.  
  - Pelanggan ini adalah penyumbang pendapatan terbesar.

- **Cluster 3 (Loyal Medium-High Value)**  
  - Monetary menengah (1.000–10.000).  
  - Pelanggan loyal dengan kontribusi signifikan.

- **Cluster 1 (Inactive Customers)**  
  - Monetary rendah, Recency tinggi.  
  - Pelanggan dengan nilai transaksi kecil dan tidak aktif.

- **Cluster 0 (Active Low-Spenders)**  
  - Monetary rendah meskipun Recency rendah.  
  - Pelanggan aktif namun pembelanjaan kecil.

---

**  3. Scatter Plot: Frequency vs Monetary**

Plot ini menunjukkan hubungan antara:
- Frequency
- Monetary


- **Cluster 2 (High-Value Customers)**  
  - Frequency dan Monetary tertinggi.  
  - Outlier transaksi besar semuanya berada pada cluster ini.  
  - Merupakan pelanggan VIP yang sangat bernilai.

- **Cluster 3 (Loyal Medium-High Value)**  
  - Frequency cukup tinggi dan Monetary menengah.  
  - Menunjukkan pelanggan loyal yang rutin berbelanja.

- **Cluster 1 (Inactive / Low-Value)**  
  - Frequency rendah, Monetary rendah.  
  - Jarang bertransaksi dan tidak memberikan kontribusi besar.

- **Cluster 0 (Active Low-Spenders)**  
  - Frequency rendah & Monetary rendah.  
  - Pelanggan baru atau pembeli bernilai rendah.
"""

# PCA
pca = PCA(n_components=2)
pca_res = pca.fit_transform(rfm_scaled_df)

rfm['pca1'] = pca_res[:,0]
rfm['pca2'] = pca_res[:,1]

sns.scatterplot(x='pca1', y='pca2', hue='Cluster', data=rfm)

"""**1. Cluster 2 — High-Value**
- Titik-titik cluster 2 berada jauh ke kanan (nilai PCA1 besar).
- Hal ini terjadi karena cluster ini memiliki **Frequency dan Monetary yang sangat tinggi**, sehingga berdampak besar pada PCA1.
- Titik-titik ekstrem di sisi kanan grafik mencerminkan pelanggan dengan nilai transaksi sangat besar.

**2. Cluster 3 — Loyal Medium-High Value**
- Terdistribusi secara diagonal ke bagian tengah grafik.
- Menunjukkan pelanggan yang aktif dan loyal, dengan kontribusi nilai transaksi menengah–tinggi.
- Persebarannya cukup stabil dan tidak terlalu ekstrem.

**3. Cluster 0 — Active Low-Spenders**
- Mengelompok pada area kiri bawah.
- Karakteristiknya adalah spending rendah dan frekuensi rendah.
- RFM mereka menghasilkan skor PCA rendah, sehingga terkonsentrasi pada bagian bawah grafik.

**4. Cluster 1 — Inactive / Lost Customers**
- Titik-titiknya sangat berdekatan di area kiri tengah.
- Memiliki Recency tinggi dan Frequency rendah.
- Karena hampir tidak bertransaksi, nilai PCA mereka cenderung kecil dan berkumpul rapat.

"""

# #Tuning Grid Search
# #from sklearn.model_selection import ParameterGrid
# from sklearn.metrics import silhouette_score


# X = rfm_scaled_df

# param_grid = {
#     "n_clusters": range(2, 11),
#     "n_init": [10, 20, 30],
#     "max_iter": [300, 500, 800],
#     "algorithm": ["lloyd"],
#     "tol": [1e-4, 1e-5]
# }
# results = []
# best_score = -1
# best_model = None
# best_params = None

# print("Proses tuning dimulai...\n")

# for k in param_grid["n_clusters"]:
#     for n_init in param_grid["n_init"]:
#         for max_iter in param_grid["max_iter"]:
#             for algo in param_grid["algorithm"]:
#                 for tol in param_grid["tol"]:

#                     kmeans = KMeans(
#                         n_clusters=k,
#                         n_init=n_init,
#                         max_iter=max_iter,
#                         algorithm=algo,
#                         tol=tol,
#                         random_state=42
#                     )

#                     labels = kmeans.fit_predict(X)

#                     score = silhouette_score(X, labels)

#                     results.append([k, n_init, max_iter, algo, tol, score])

#                     print(f"K={k}, n_init={n_init}, iter={max_iter}, tol={tol} → Silhouette: {score:.4f}")

#                     if score > best_score:
#                         best_score = score
#                         best_model = kmeans
#                         best_params = {
#                             "n_clusters": k,
#                             "n_init": n_init,
#                             "max_iter": max_iter,
#                             "algorithm": algo,
#                             "tol": tol
#                         }

import joblib
joblib.dump(kmeans, 'kmeans_rfm_model.pkl')
joblib.dump(scaler, 'rfm_scaler.pkl')

# from google.colab import files

# files.download('kmeans_rfm_model.pkl')
# files.download('rfm_scaler.pkl')

import pandas as pd
import os

# Pastikan Anda sudah mengupload 'OnlineRetail.csv' ke folder (ikon folder di kiri Colab)
file_name = 'OnlineRetail.csv'

if os.path.exists(file_name):
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    print("Dataset berhasil dimuat!")
    print(f"Jumlah baris awal: {len(df)}")
else:
    print("Peringatan: File OnlineRetail.csv belum diupload ke Colab!")

# 1. Menghapus CustomerID yang kosong (wajib untuk segmentasi)
df_clean = df.dropna(subset=['CustomerID'])

# 2. Menghapus Quantity dan UnitPrice yang bernilai 0 atau negatif
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

# 3. Membuat kolom TotalPrice
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"Jumlah baris setelah dibersihkan: {len(df_clean)}")
df_clean.head()

script_content = """
import pandas as pd
import os

def run_preprocessing(input_path):
    # Load data
    df = pd.read_csv(input_path, encoding='ISO-8859-1')

    # Cleaning
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Save hasil bersih
    output_name = "OnlineRetail_preprocessed.csv"
    df.to_csv(output_name, index=False)
    print(f"Automasi Berhasil! File tersimpan: {output_name}")

if __name__ == "__main__":
    run_preprocessing('OnlineRetail.csv')
"""

with open('automate_Ahul.py', 'w') as f:
    f.write(script_content)

print("File 'automate_Ahul.py' telah dibuat di folder Colab.")

# from google.colab import files

# Download script automasi
# files.download('automate_Ahul.py')

# Download data yang sudah bersih (untuk jaga-jaga)
df_clean.to_csv('OnlineRetail_preprocessed.csv', index=False)
# files.download('OnlineRetail_preprocessed.csv')

# !pip install mlflow dagshub

import mlflow
import dagshub
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Hubungkan ke DagsHub
dagshub.init(repo_owner='IslahulHadi', repo_name='Retail-OpsML-Ahul', mlflow=True)

# 1. Load Data
df = pd.read_csv('OnlineRetail_preprocessed.csv')
# Gunakan fitur RFM sederhana (Quantity & UnitPrice)
X = df[['Quantity', 'UnitPrice']]

# 2. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. MLflow Tracking
with mlflow.start_run(run_name="KMeans_Segmentation"):
    # Gunakan 3 cluster sebagai contoh awal
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    # Simpan Parameter & Model ke DagsHub
    mlflow.log_param("n_clusters", 3)
    mlflow.sklearn.log_model(kmeans, "retail_segmentation_model")

    print("Berhasil! Cek tab 'Experiments' di DagsHub Anda.")

from sklearn.metrics import silhouette_score

# ... (kode load data dan scaling yang tadi) ...

with mlflow.start_run(run_name="KMeans_Segmentation_With_Metrics"):
    # Training
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    # HITUNG METRIK (Ini yang membuat Chart muncul)
    score = silhouette_score(X_scaled, kmeans.labels_)

    # Log Parameter
    mlflow.log_param("n_clusters", 3)

    # LOG METRIK KE MLFLOW
    mlflow.log_metric("silhouette_score", score)

    # Log Model
    mlflow.sklearn.log_model(kmeans, "retail_segmentation_model")

    print(f"Berhasil! Score: {score}. Cek tab 'Charts' di DagsHub sekarang.")

# !pip install pipreqs

# !pipreqs . --force

