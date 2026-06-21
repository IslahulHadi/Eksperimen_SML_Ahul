# Submission-MA-5 — Customer Segmentation dengan K-Means

**Oleh:** Ahul (IslahulHadi)

## Deskripsi Proyek
Proyek ini merupakan implementasi **Customer Segmentation** menggunakan algoritma **K-Means Clustering** pada dataset Online Retail. Segmentasi dilakukan berdasarkan analisis **RFM (Recency, Frequency, Monetary)** untuk mengidentifikasi kelompok pelanggan.

## Struktur Proyek
```
├── membangun_model/
│   ├── modelling.py                  # Script training model K-Means
│   ├── modelling_tuning.py           # Script modelling + tuning + MLflow
│   ├── online_retail_preprocessing.py # Script preprocessing data
│   ├── requirements.txt              # Dependencies
│   ├── DagHub.txt                    # Link DagsHub repository
│   ├── screenshoot_artifak.png       # Bukti artefak MLflow
│   └── screenshoot_dashboard.png     # Bukti dashboard MLflow
├── monitoring_dan_logging/
│   ├── 1.bukti_serving.png           # Bukti model serving
│   ├── 2.prometheus.yml              # Konfigurasi Prometheus
│   ├── 3.prometheus_exporter.py      # Prometheus exporter + Flask server
│   ├── 4.bukti monitoring Prometheus/ # Screenshot monitoring Prometheus
│   ├── 5.bukti monitoring Grafana/   # Screenshot monitoring Grafana
│   ├── 6.bukti alerting Grafana/     # Screenshot alerting Grafana
│   └── 7.inference.py               # Script inference/prediksi
├── Eksperimen_SML_Ahul.txt           # Link repo eksperimen
├── Workflow-CI.txt                   # Link repo Workflow CI/CD
└── README.md
```

## Teknologi
- Python, Pandas, Scikit-learn
- K-Means Clustering, PCA, StandardScaler
- MLflow + DagsHub (tracking & monitoring)
- Prometheus + Grafana (monitoring & alerting)
- Flask + prometheus_client (model serving & metrics)

## Link Terkait
- **GitHub Eksperimen:** [Eksperimen_SML_Ahul](https://github.com/IslahulHadi/Eksperimen_SML_Ahul)
- **Workflow CI/CD:** [WORKFLOW-CL](https://github.com/IslahulHadi/WORKFLOW-CL)
- **DagsHub:** [Retail-OpsML-Ahul](https://dagshub.com/IslahulHadi/Retail-OpsML-Ahul)
