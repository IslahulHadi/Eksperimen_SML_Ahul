# Submission-MA-5 — Customer Segmentation dengan K-Means

**Oleh:** Ahul (IslahulHadi)

## Deskripsi Proyek
Proyek ini merupakan implementasi **Customer Segmentation** menggunakan algoritma **K-Means Clustering** pada dataset Online Retail. Segmentasi dilakukan berdasarkan analisis **RFM (Recency, Frequency, Monetary)** untuk mengidentifikasi kelompok pelanggan.

## Struktur Proyek
```
├── membangun_model/
│   ├── modelling.py                  # Script modelling utama
│   ├── modelling_tuning.py           # Script modelling + tuning + MLflow
│   ├── online_retail_preprocessing.py # Script preprocessing data
│   ├── requirements.txt              # Dependencies
│   ├── DagHub.txt                    # Link DagsHub repository
│   ├── screenshoot_artifak.png       # Bukti artefak MLflow
│   └── screenshoot_dashboard.png     # Bukti dashboard MLflow
├── monitoring_dan_logging/
│   ├── inference.py                  # Script inference/prediksi
│   ├── monitoring.txt                # Keterangan monitoring
│   └── bukti_serving.png             # Bukti model serving
├── Eksperimen_SML_Ahul.txt           # Link repo eksperimen
├── Workflow-CI.txt                   # Link repo Workflow CI/CD
└── README.md
```

## Teknologi
- Python, Pandas, Scikit-learn
- K-Means Clustering, PCA
- MLflow + DagsHub (tracking & monitoring)

## Link Terkait
- **GitHub Eksperimen:** [Eksperimen_SML_Ahul](https://github.com/IslahulHadi/Eksperimen_SML_Ahul)
- **Workflow CI/CD:** [WORKFLOW-CL](https://github.com/IslahulHadi/WORKFLOW-CL)
- **DagsHub:** [Retail-OpsML-Ahul](https://dagshub.com/IslahulHadi/Retail-OpsML-Ahul)
