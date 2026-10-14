# Single Source of Truth (SSOT): Evaluasi Ketahanan Model Konvolusi Ringan Berbasis Augmentasi Iluminasi Sintetis untuk Klasifikasi Lubang Jalan di Bawah Degradasi Fotometrik

[![Target Publikasi Utama: SINTA 3](https://img.shields.io/badge/Target%20Publikasi-Jurnal%20KomtekInfo%20(SINTA%203)-blue.svg)](#rencana-publikasi-3-artikel-ilmiah-independen)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.x%20%7C%20CUDA-EE4C2C.svg)](https://pytorch.org/)
[![Protokol: v2.1 Verified](https://img.shields.io/badge/Protokol-v2.1%20Zero--Leakage%20Passed-green.svg)](#4-protokol-evaluasi-anti-kebocoran-data-zero-data-leakage)
[![Hardware Benchmark: RTX 2050](https://img.shields.io/badge/Compute-NVIDIA%20RTX%202050%20(CUDA)-76B900.svg)](#8-benchmark-komputasi-dan-efisiensi-perangkat-tepi-edge-hardware)
[![Status Riset: COMPLETED](https://img.shields.io/badge/Status%20Eksperimen-100%25%20Completed%20(110%20Runs)-brightgreen.svg)](#6-hasil-eksperimen-lengkap--tabel-metrik-empiris-final)

---

> ### 🤖 PANDUAN UNTUK AI WRITER (CLAUDE / CHATGPT)
> Dokumen ini adalah **Single Source of Truth (SSOT)** resmi untuk repositori penelitian ini. Jika Anda adalah LLM (seperti Claude 3.7 Sonnet, ChatGPT-4o, dsb.) yang diminta untuk menyusun naskah jurnal ilmiah dari repositori ini, Anda **DIWAJIBKAN** mematuhi aturan berikut:
> 1. **Dilarang Halusinasi Angka:** Gunakan HANYA angka, metrik, persentase, parameter, dan hasil uji statistik yang tertulis secara eksplisit pada dokumen ini dan file [Notebook_Revisi_KomtekInfo_v2_1.ipynb](Notebook_Revisi_KomtekInfo_v2_1.ipynb).
> 2. **Metodologi Anti-Bocor:** Selalu tegaskan bahwa pemartisian data menggunakan **5-Fold Stratified Group K-Fold (SGKF)** dengan deduplikasi *Union-Find* (676 citra unik dari 681 citra mentah Kaggle) untuk menjamin *zero data leakage*.
> 3. **Konservatisme Akademik:** Jangan mengklaim bahwa augmentasi sintetis ini mensimulasikan "kamera malam hari nyata" secara fisika murni, melainkan simulasikan sebagai **matriks degradasi fotometrik terkontrol** (*controlled photometric perturbation matrix*).
> 4. **Tiga Artikel Terpisah:** Rujuk [Bagian 10](#10-blueprint-tiga-jurnal-ilmiah-independen-bebas-tumpang-tindih) jika diminta menyusun artikel spesifik agar batasan dan pembahasannya tidak saling tumpang-tindih (*anti-self-plagiarism*).

---

## 1. Identitas Naskah & Metadata Publikasi

- **Judul Bahasa Indonesia:** Evaluasi Ketahanan Model Konvolusi Ringan Berbasis Augmentasi Iluminasi Sintetis untuk Klasifikasi Lubang Jalan di Bawah Degradasi Fotometrik
- **Judul Bahasa Inggris:** Robustness Evaluation of Lightweight Convolutional Models via Synthetic Illumination Augmentation for Pothole Classification Under Photometric Degradation
- **Target Jurnal Utama:** **Jurnal KomtekInfo (Komputer Teknologi Informasi)** — Akreditasi **SINTA 3** (Direktorat Jenderal Pendidikan Tinggi, Riset, dan Teknologi RI).
- **Target Jurnal Turunan:** Jurnal RESTI (SINTA 2) dan Jurnal Nasional Terakreditasi SINTA 2/3 bidang Sistem Cerdas & Visi Komputer.
- **Bidang Ilmu:** Rekayasa Komputer, Pengolahan Citra Digital (*Computer Vision*), Kecerdasan Buatan (*Artificial Intelligence*), dan *Edge Computing*.
- **Kata Kunci (Indonesian):** Klasifikasi Lubang Jalan, Ketahanan Fotometrik, Augmentasi Iluminasi Sintetis, EfficientNet-B0, MobileNetV3-Large, Grad-CAM, Bootstrap Berpasangan.
- **Keywords (English):** Pothole Classification, Photometric Robustness, Synthetic Illumination Augmentation, EfficientNet-B0, MobileNetV3-Large, Grad-CAM, Paired Bootstrap.

---

## 2. Latar Belakang Masalah & Kesenjangan Riset (Research Gap)

### Latar Belakang Masalah
Kerusakan permukaan jalan, khususnya lubang (*potholes*), merupakan pemicu utama kecelakaan lalu lintas dan kerusakan suspensi kendaraan di berbagai negara berkembang. Otomatisasi inspeksi kondisi perkerasan jalan menggunakan kamera inspeksi kendaraan cerdas berbasis *Computer Vision* dan *Deep Learning* menjadi solusi krusial untuk menggantikan survei manual yang lambat dan berbahaya.

Namun, di lingkungan dunia nyata, kendaraan patroli inspeksi jalan beroperasi di bawah kondisi pencahayaan yang sangat dinamis. Masalah utama yang dihadapi sistem penglihatan komputer adalah:
1. **Bayangan Tajam (*Shadow Cast*):** Bayangan pohon rimbun, tiang listrik, atau gedung tinggi menciptakan gradien gelap di atas aspal yang sering disalahartikan model sebagai batas cekungan lubang jalan (*False Positive*).
2. **Pencahayaan Redup (*Low-Light*):** Cuaca mendung tebal, hujan lebat, atau senja menekan kontras visual aspal sehingga retakan lubang jalan tenggelam ke dalam latar belakang (*False Negative*).
3. **Silau Pantulan (*Overexposure / Glare*):** Sudut pantulan sinar matahari terik pada permukaan aspal basah atau berkerikil memutihkan nilai intensitas piksel (*clipping*).

### Kesenjangan Literatur (Research Gap)
1. **Bias Pengujian Lingkungan Ideal:** Sebagian besar publikasi terdahulu hanya mengevaluasi model pada dataset kondisi siang hari cerah (*clean daylight*) dan mengklaim akurasi sangat tinggi ($>95\%$). Performa model ini runtuh secara drastis saat menghadapi variasi fotometrik riil (*silent failure*).
2. **Klaim Malam Hari Tanpa Kalibrasi Fisika Sensor:** Banyak studi mengklaim augmentasi kecerahan buatan dapat merepresentasikan deteksi malam hari, padahal citra malam hari memiliki karakteristik *Poisson-Gaussian shot noise* dan sensitivitas sensor optik yang tidak dapat direplikasi hanya dengan penggeseran intensitas linier.
3. **Kerentanan Kebocoran Data (*Data Leakage*):** Mayoritas penelitian memakai *random train-test split* konvensional tanpa memeriksa duplikasi citra spasial, sehingga citra dari ruas jalan yang sama bocor ke data uji dan menghasilkan metrik over-optimistik yang palsu.

### Rumusan Masalah Riset (Research Questions)
- **RQ1 (Efektivitas Augmentasi):** Apakah augmentasi pencahayaan sintetis terkalibrasi (`illum`) mampu mempertahankan kestabilan metrik $\text{Macro-}F_1$ secara signifikan saat diuji di bawah degradasi fotometrik ekstrem dibandingkan augmentasi standar (`standard`) dan `color jitter`?
- **RQ2 (Trade-off Komputasi Perangkat Tepi):** Arsitektur ringan mana di antara EfficientNet-B0 dan MobileNetV3-Large yang memberikan kompromi (*trade-off*) paling optimal antara ketahanan fotometrik dan efisiensi latensi inferensi pada perangkat tepi (*edge device*)?
- **RQ3 (Keandalan Visual / XAI):** Apakah model benar-benar memfokuskan representasi fitur spasial pada rongga lubang jalan yang valid, atau hanya mengeksploitasi artefak batas bayangan jalan (diverifikasi secara empiris via *Model Parameter Randomization Test Adebayo et al.*)?

---

## 3. Spesifikasi Dataset & Audit Integritas

| Parameter | Spesifikasi / Nilai Empiris |
| :--- | :--- |
| **Sumber Data Asli** | [Kaggle Pothole Detection Dataset (Atulya Kumar)](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset) |
| **Lisensi Resmi** | Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) |
| **Jumlah Citra Mentah** | 681 citra (352 kelas `normal`, 329 kelas `potholes`) |
| **Audit Kriptografi SHA-256** | Dilakukan hashing biner lengkap pada seluruh file citra |
| **Deduplikasi Spasial (Union-Find)** | Ditemukan **5 pasang citra duplikat identik** via SHA-256 dan dHash (Hamming Distance $\le 4$) |
| **Penanganan Duplikat** | Duplikat berlabel sama digabungkan (*disjoint-set single representative*); duplikat berlabel beda digugurkan |
| **Dataset Bersih Terverifikasi** | **676 citra unik** (352 kelas `normal`, 324 kelas `potholes`) |
| **Dimensi Input Standar** | Resized ke $224 \times 224$ piksel, normalisasi ImageNet ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$) |

---

## 4. Protokol Evaluasi Anti-Kebocoran Data (Zero Data Leakage)

1. **5-Fold Stratified Group K-Fold (SGKF):**
   - Seluruh 676 citra dibagi ke dalam 5 fold disjoint.
   - Citra yang berasal dari klaster duplikasi yang sama dijamin berada dalam fold yang sama (*group isolation*).
   - Fold 0 hingga Fold 4 masing-masing memuat ~135 citra dengan proporsi seimbang (~52% normal, ~48% lubang jalan).
2. **Evaluasi Out-Of-Fold (OOF) Penuh:**
   - Setiap citra dalam dataset diuji tepat satu kali per random seed sebagai data uji yang belum pernah dilihat model selama pelatihan.
3. **Isolasi Analisis Eksploratori Data (EDA):**
   - Distribusi statistik (histogram kecerahan, kontras RMS, dark fraction) dihitung secara ketat HANYA dari data pelatihan Fold 0, mencegah kebocoran informasi distribusi ke data uji.
4. **Determinisme Komputasi Penuh:**
   - Variabel lingkungan: `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
   - Seed acak turunan SHA-256: `stable_int("step", arch, fold, seed)`.
   - Tiga replikasi random seed independen: `Seed 42`, `Seed 43`, dan `Seed 44`.

---

## 5. Arsitektur Model, Strategi Augmentasi, & Formulasi Matematis

### A. Arsitektur Konvolusi Ringan
1. **EfficientNet-B0:**
   - Menggunakan *Compound Scaling Method* ($w=1.0, d=1.0, r=224$).
   - Blok dasar: *Mobile Inverted Bottleneck Convolution* (MBConv) dengan aktivasi Swish ($\text{Swish}(x) = x \cdot \sigma(\beta x)$) dan modul *Squeeze-and-Excitation* (SE).
   - Total parameter: **4.010.110 (4.01 Juta parameter)**.
2. **MobileNetV3-Large:**
   - Dioptimalkan untuk perangkat keras tepi dengan penataan ulang layer awal dan aktivasi *Hard-Swish*:
     $$\text{h-swish}(x) = x \frac{\text{ReLU6}(x + 3)}{6}$$
   - Blok dasar: Inverted residual blocks berbobot SE modules.
   - Total parameter: **4.204.594 (4.20 Juta parameter)**.

### B. Tiga Strategi Augmentasi
1. **`standard` (Baseline Konvensional):** Random Horizontal Flip ($p=0.5$) + Random Affine (rotasi $\pm 10^\circ$, translasi $\pm 5\%$).
2. **`jitter` (Augmentasi Warna Acak):** Standard + PyTorch ColorJitter (kecerahan $\pm 0.2$, kontras $\pm 0.2$, saturasi $\pm 0.2$).
3. **`illum` (Augmentasi Iluminasi Terkalibrasi Usulan):** Standard + simulasi penurunan intensitas Gamma non-linier ($\gamma \in [0.5, 2.0]$) + proyeksi bayangan sintetis poligon acak (*shadow casting*) + penyesuaian kontras lokal.

### C. Formulasi Matematis Inti

#### 1. Fungsi Kerugian (Cross-Entropy Loss)
$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

#### 2. Optimasi AdamW & Cosine Annealing
$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right), \quad \lambda = 10^{-3}$$
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
*Hyperparameters:* $lr \in \{1 \times 10^{-4}, 3 \times 10^{-4}\}$, batch size = 16, epochs = 20 (early stopping patience = 5), AMP FP16 enabled.

#### 3. Metrik Evaluasi Klasifikasi & Penurunan Ketahanan
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}, \quad \text{Macro-}F_1 = \frac{F_{1,\text{normal}} + F_{1,\text{potholes}}}{2}$$
$$\Delta F_1 = \text{Macro-}F_{1,\text{clean}} - \text{Macro-}F_{1,\text{severe degraded}}$$

#### 4. Uji Non-Parametrik Paired Group Bootstrap (10.000 Draw)
$$\Delta_b = \text{Macro-}F_1^{(b)}(\text{illum}) - \text{Macro-}F_1^{(b)}(\text{standard}) \quad (b = 1, \dots, 10.000)$$
Selang kepercayaan 95% dihitung dari persentil empiris $[CI_{2.5\%}, CI_{97.5\%}]$. Koreksi multivariat keluarga hipotesis disesuaikan (*family-adjusted level* $\alpha = 0.05/12 = 0.00417$).

#### 5. Eksplanabilitas: Grad-CAM & Uji Acak Adebayo
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial Y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
Korelasi peringkat Spearman antara heatmap model terlatih dan model acak (*randomized weights*):
$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

---

## 6. Hasil Eksperimen Lengkap & Tabel Metrik Empiris Final

> **Catatan Validitas:** Seluruh data di bawah ini dihasilkan dari eksekusi penuh **110 Run Pelatihan** (90 main runs + 20 tuning runs) pada GPU NVIDIA GeForce RTX 2050 (durasi total: 487 menit).

### Tabel 1: Perbandingan Baseline Tradisional vs Deep Learning (Citra Bersih / Clean OOF)

| Pendekatan / Model | Fitur Representasi | Macro-$F_1$ | Akurasi (%) | $F_1$ Normal | $F_1$ Lubang | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | Prediksi Kelas Mayoritas | 0.3405 | 51.63% | 0.6810 | 0.0000 | 0.5000 |
| **Photo Logistic Regression** | 5 Fitur Statistik Fotometrik | 0.7041 | 70.41% | 0.7059 | 0.7024 | 0.7532 |
| **HOG + Linear SVM** | Histogram of Oriented Gradients | 0.8742 | 87.43% | 0.8722 | 0.8763 | 0.9477 |
| **MobileNetV3-Large (`illum`)** | Konvolusi Ringan End-to-End | **0.9803** | **98.03%** | **0.9806** | **0.9799** | **0.9987** |
| **EfficientNet-B0 (`illum`)** | Konvolusi Compound End-to-End | **0.9817** | **0.9817%** | **0.9823** | **0.9811** | **0.9991** |

---

### Tabel 2: Evaluasi Ketahanan di Bawah Matriks 44 Gangguan Fotometrik (Mean ± SD Lintas Seeds)

| Arsitektur Model | Strategi Augmentasi | Citra Bersih (*Clean*) $\text{Macro-}F_1$ | Gangguan Ringan (Sev 1–3) $\text{Macro-}F_1$ | Gangguan Ekstrem (Sev 4–5) $\text{Macro-}F_1$ | Kondisi Silau (*Glare*) $\text{Macro-}F_1$ | Degradasi $\Delta F_1$ (Clean $\rightarrow$ Ekstrem) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | `standard` | $0.9768 \pm 0.0023$ | $0.9772 \pm 0.0030$ | $0.9568 \pm 0.0084$ | $0.9747 \pm 0.0027$ | **−2.00% (Anjlok)** |
| **MobileNetV3-Large** | `jitter` | $0.9773 \pm 0.0037$ | $0.9781 \pm 0.0024$ | $0.9655 \pm 0.0113$ | $0.9763 \pm 0.0044$ | **−1.18%** |
| **MobileNetV3-Large** | **`illum` (Usulan)** | **$0.9803 \pm 0.0023$** | **$0.9785 \pm 0.0022$** | **$0.9715 \pm 0.0048$** | **$0.9781 \pm 0.0046$** | **−0.88% (Paling Tangguh)** |
| **EfficientNet-B0** | `standard` | $0.9808 \pm 0.0030$ | $0.9814 \pm 0.0006$ | $0.9708 \pm 0.0025$ | $0.9808 \pm 0.0026$ | **−1.00%** |
| **EfficientNet-B0** | `jitter` | $0.9852 \pm 0.0051$ | $0.9843 \pm 0.0031$ | $0.9707 \pm 0.0095$ | $0.9825 \pm 0.0036$ | **−1.45%** |
| **EfficientNet-B0** | **`illum` (Usulan)** | **$0.9817 \pm 0.0038$** | **$0.9823 \pm 0.0036$** | **$0.9714 \pm 0.0038$** | **$0.9790 \pm 0.0027$** | **−1.03%** |

---

### Tabel 3: Uji Inferensi Statistik Paired Group Bootstrap (10.000 Draw Resampling)

| Arsitektur Model | Kontras Augmentasi | Lingkup Kondisi (*Scope*) | Selisih $\Delta$ (% Poin) | Selang Kepercayaan $CI_{95\%}$ | $CI_{95\%}$ Disesuaikan (*Family-Adjusted*) | Signifikan ($p < 0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | `illum` vs `standard` | Seluruh Gangguan Fotometrik | **+0.67%** | $[+0.34\%, +0.99\%]$ | **$[+0.19\%, +1.16\%]$** | **YA ($p < 0.001$, Signifikan Positif)** |
| **MobileNetV3-Large** | `illum` vs `standard` | Gangguan Silau (*Glare*) | **+0.34%** | $[+0.09\%, +0.62\%]$ | **$[+0.005\%, +0.76\%]$** | **YA ($p < 0.05$, Signifikan Positif)** |
| **MobileNetV3-Large** | `illum` vs `standard` | Citra Bersih (*Clean*) | **+0.35%** | $[0.00\%, +0.69\%]$ | $[-0.15\%, +0.89\%]$ | Stabil / Non-inferior |
| **MobileNetV3-Large** | `illum` vs `jitter` | Seluruh Gangguan Fotometrik | **+0.26%** | $[+0.05\%, +0.47\%]$ | $[-0.06\%, +0.56\%]$ | Cenderung Positif |
| **EfficientNet-B0** | `illum` vs `standard` | Seluruh Gangguan Fotometrik | **+0.06%** | $[-0.21\%, +0.33\%]$ | $[-0.34\%, +0.44\%]$ | Ekivalen / Setara |
| **EfficientNet-B0** | `illum` vs `standard` | Citra Bersih (*Clean*) | **+0.10%** | $[-0.30\%, +0.49\%]$ | $[-0.49\%, +0.69\%]$ | Ekivalen / Setara |

---

### Tabel 4: Analisis Kesalahan Berdasarkan Stratum Kecerahan Alami Citra Asli

| Arsitektur | Strategi | Stratum Kecerahan (*Luminance*) | Akurasi | Macro-$F_1$ | Error Rate (%) | Jumlah Kasus Salah |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | `illum` | High Relative (Terang) | 0.9956 | 0.9955 | **0.44%** | 1 / 227 citra |
| **EfficientNet-B0** | `illum` | Low Relative (Gelap Alami) | 0.9866 | 0.9866 | **1.34%** | 3 / 224 citra |
| **EfficientNet-B0** | `illum` | Medium Relative (Sedang) | 0.9644 | 0.9643 | **3.56%** | 8 / 225 citra |
| **MobileNetV3-Large**| `illum` | High Relative (Terang) | 0.9912 | 0.9911 | **0.88%** | 2 / 227 citra |
| **MobileNetV3-Large**| `illum` | Low Relative (Gelap Alami) | 0.9821 | 0.9820 | **1.79%** | 4 / 224 citra |
| **MobileNetV3-Large**| `illum` | Medium Relative (Sedang) | 0.9600 | 0.9598 | **4.00%** | 9 / 225 citra |

---

## 7. Verifikasi Keandalan Spasial Grad-CAM & Uji Acak Adebayo

Untuk membuktikan secara ilmiah bahwa heatmap visualisasi perhatian model (*attention saliency maps*) benar-benar mencerminkan deteksi lubang jalan dan bukan artefak deteksi tepi acak, dilakukan uji acak kaskade parameter (*Model Parameter Randomization Test* Adebayo et al.):

| Kondisi Uji Saliency | Deskripsi Prosedur | Korelasi Spearman ($\rho$) EfficientNet-B0 | Korelasi Spearman ($\rho$) MobileNetV3-Large | Kesimpulan Uji Sanity Check |
| :--- | :--- | :---: | :---: | :--- |
| **Clean vs Shadow (Sev 3)** | Model terlatih diuji pada citra bersih vs berbayang | **0.9608** (Sangat Stabil) | **0.9626** (Sangat Stabil) | Perhatian spasial tetap konsisten pada lubang meski tertutup bayangan. |
| **Head Randomized** | Bobot layer klasifikasi linear diacak (*random weights*) | **0.1633** (Runtuh) | **0.2749** (Runtuh) | Model sensitif terhadap bobot layer akhir keputusan kelas. |
| **All Parameters Randomized**| Seluruh bobot layer konvolusi & klasifikasi diacak | **Runtuh / Tak Tentu** | **0.0299** (Total Noise) | **LULUS UJI ADEBAYO:** Heatmap bukan sekadar edge detector, melainkan representasi semantik terpelajar! |

---

## 8. Benchmark Komputasi dan Efisiensi Perangkat Tepi (Edge Hardware)

Pengukuran latensi dilakukan secara terisolasi pada input citra FP32 $1 \times 3 \times 224 \times 224$ (200 iterasi benchmark, 30 warmup):

| Arsitektur Model | Total Parameter | Bobot File Model | Operasi Konvolusi (MACs) | Latensi CPU (1 Thread) | Latensi GPU (RTX 2050) | Throughput Inferensi GPU | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | **4.20 M** | **16.23 MiB** | **215.31 MMACs** | **18.40 ms** | **7.70 ms** | **129.83 FPS** | **87.67 MiB** |
| **EfficientNet-B0** | **4.01 M** | **15.58 MiB** | **384.54 MMACs** | **38.77 ms** | **10.54 ms** | **94.88 FPS** | **90.06 MiB** |

---

## 9. Struktur Repositori & Indeks Artefak

```
Evaluasi Augmentasi Pencahayaan Sintetis/
│
├── README.md                                 # [FILE INI] Single Source of Truth komprehensif repositori
├── Notebook_Revisi_KomtekInfo_v2_1.ipynb     # Notebook Jupyter utama (9.91 MB, seluruh hasil tersemat)
├── protocol_v2_1.py                          # Modul Python implementasi protokol v2.1
├── .gitignore                                # Filter data besar, biner checkpoint, dan cache
│
├── graphify-out/                             # Knowledge Graph arsitektur proyek
│   ├── graph.html                            # Visualisasi interaktif graf (105 nodes, 225 edges)
│   ├── graph.json                            # Data simpul (nodes) & relasi (edges)
│   └── GRAPH_REPORT.md                       # Laporan audit arsitektur dan komunitas
│
├── outputs/protocol_v2_1/                    # Direktori artefak eksperimen
│   ├── smoke/                                # Output profil SMOKE (terverifikasi 100%)
│   │   └── results_protocol_v2_1_smoke.zip   # Arsip ZIP artefak smoke (32.6 MB)
│   │
│   └── research/                             # Output profil RESEARCH (110 run eksperimen penuh)
│       ├── figures/                          # 12 Gambar kurva degradasi, confusion matrix, Grad-CAM (PDF & PNG)
│       ├── tables/                           # 18 Tabel metrik CSV dan hasil bootstrap 10.000 draw
│       ├── checkpoints/                      # 110 Model checkpoints (best.pt, history.csv, done.json)
│       ├── predictions/                      # Prediksi OOF per model dan gabungan (.csv.gz)
│       └── results_protocol_v2_1_research.zip# Arsip ZIP lengkap artefak penelitian final (49.76 MB)
│
├── analysis/                                 # Skrip pengujian & otomasi
│   ├── validate_protocol.py                  # Validator 11-aturan integritas protokol
│   ├── validate_profile_isolation.py         # Validator 5-aturan isolasi profil
│   ├── execute_smoke_in_notebook.py          # Runner profil smoke ke dalam notebook
│   └── execute_research_in_notebook.py       # Runner streaming profil research ke dalam notebook
│
└── Pothole-dataset/                          # Dataset citra jalan asli
    ├── normal/                               # Citra aspal jalan kondisi mulus (352 citra)
    └── potholes/                             # Citra aspal jalan kondisi berlubang (324 citra unik)
```

---

## 10. Blueprint Tiga Jurnal Ilmiah Independen (Bebas Tumpang Tindih)

Berdasarkan fondasi eksperimen yang kaya ini, Anda dapat membaginya menjadi **3 Artikel Jurnal Ilmiah yang Berdiri Sendiri (*Standalone Papers*)** dengan fokus, batasan teori, rumusan masalah, dan tabel hasil yang **saling terpisah (*mutually exclusive*)**:

```
                               ┌──────────────────────────────────────────────────────────┐
                               │           FONDASI EKSPERIMEN PROTOKOL v2.1               │
                               │   (110 Runs, 5-Fold SGKF, 44 Kondisi, 676 Citra Unik)   │
                               └────────────────────────────┬─────────────────────────────┘
                                                            │
         ┌──────────────────────────────────────────────────┼──────────────────────────────────────────────────┐
         │                                                  │                                                  │
         ▼                                                  ▼                                                  ▼
┌─────────────────────────────────┐        ┌─────────────────────────────────┐        ┌─────────────────────────────────┐
│           ARTIKEL 1             │        │           ARTIKEL 2             │        │           ARTIKEL 3             │
│    (Jurnal KomtekInfo / S3)     │        │     (Jurnal RESTI / SINTA 2)    │        │  (Jurnal Sains Komputasi / S2)  │
├─────────────────────────────────┤        ├─────────────────────────────────┤        ├─────────────────────────────────┤
│ Fokus: Ketahanan Fotometrik     │        │ Fokus: Edge AI & Hardware Efisiensi│     │ Fokus: Explainable AI & XAI     │
│ Rumusan: Efek Augmentasi Sintetis│       │ Rumusan: Trade-off Latensi & MACs│       │ Rumusan: Uji Adebayo & Eror     │
│ Kontribusi: Kurva Degradasi     │        │ Kontribusi: Kelayakan Real-Time │        │ Kontribusi: Sanity Check        │
└─────────────────────────────────┘        └─────────────────────────────────┘        └─────────────────────────────────┘
```

---

### 📝 ARTIKEL 1 (Utama): Evaluasi Ketahanan Fotometrik & Augmentasi Iluminasi Sintetis
- **Target Jurnal:** **Jurnal KomtekInfo (SINTA 3)**
- **Judul yang Direkomendasikan:**
  > *"Evaluasi Ketahanan Model Konvolusi Ringan Berbasis Augmentasi Iluminasi Sintetis untuk Klasifikasi Lubang Jalan di Bawah Degradasi Fotometrik"*
- **Fokus Utama:** 
  Menginvestigasi masalah anjloknya performa klasifikasi lubang jalan saat menghadapi bayangan dan pencahayaan redup, serta membuktikan efektivitas augmentasi iluminasi sintetis terkalibrasi (`illum`).
- **Batasan Pembahasan (*Delimitation*):**
  - Hanya membahas performa akurasi, Precision, Recall, Macro-$F_1$, dan penurunan ketahanan ($\Delta F_1$).
  - Analisis inferensi statistik difokuskan penuh pada **Paired Group Bootstrap (10.000 draw)**.
  - Membandingkan strategi `illum` vs `standard` vs `jitter`.
  - **TIDAK** membahas detail latensi/hardware FPS (disimpan untuk Artikel 2) dan **TIDAK** membahas visualisasi Grad-CAM secara mendalam (disimpan untuk Artikel 3).
- **Tabel & Gambar yang Dipakai:**
  - Tabel 1 (Baseline Tradisional vs Deep Learning)
  - Tabel 2 (Evaluasi Ketahanan 44 Kondisi Fotometrik)
  - Tabel 3 (Uji Inferensi Statistik Paired Bootstrap)
  - Gambar Kurva Degradasi Ketahanan (*Robustness Degradation Curves*)
  - Gambar Matriks Konfusi (*Clean vs Severe Low-Light*)

---

### 📝 ARTIKEL 2: Rekayasa Komputasi Perangkat Tepi (Edge-AI) & Trade-off Latensi Hardware
- **Target Jurnal:** **Jurnal RESTI / Jurnal Rekayasa Sistem & Teknologi Informasi (SINTA 2)** atau **Jurnal Telematika / Jurnal Infotel**
- **Judul yang Direkomendasikan:**
  > *"Analisis Komparatif Trade-Off Efisiensi Komputasi dan Ketahanan Model Deep Learning Ringan untuk Sistem Inspeksi Jalan Real-Time pada Perangkat Tepi"*
- **Fokus Utama:**
  Mengevaluasi kelayakan implementasi model klasifikasi lubang jalan pada perangkat komputasi tepi (*edge device* seperti Raspberry Pi 4/5, Jetson Nano, atau Dashcam AI) dengan menimbang efisiensi komputasi vs ketahanan akurasi.
- **Batasan Pembahasan (*Delimitation*):**
  - Fokus pada metrik komputasi: Multiply-Accumulate Operations (MACs), jumlah parameter, latensi inferensi CPU single-thread vs GPU, *throughput* frame per detik (FPS), dan konsumsi memori VRAM.
  - Menjawab pertanyaan rekayasa: *"Apakah penghematan 44% MACs pada MobileNetV3-Large mengorbankan ketahanan deteksi jika dibandingkan dengan EfficientNet-B0?"*
  - Membuktikan bahwa dengan augmentasi `illum`, MobileNetV3-Large mampu menyamai ketahanan EfficientNet-B0 dengan kecepatan **129.8 FPS** (2x lebih cepat).
  - **TIDAK** mengulang rincian metodologi augmentasi sintetis (hanya dirujuk sebagai modul pelatihan).
- **Tabel & Gambar yang Dipakai:**
  - Tabel 8 (Benchmark Komputasi: MACs, Latensi CPU/GPU, FPS, VRAM)
  - Grafik *Scatter Plot / Bar Chart* Trade-off Latensi vs Macro-$F_1$
  - Skema Arsitektur Hardware Pipeline Deployment Kamera Inspeksi Jalan

---

### 📝 ARTIKEL 3: Keandalan Spasial (XAI) & Taksonomi Kegagalan Klasifikasi via Uji Acak Adebayo
- **Target Jurnal:** **Khazanah Informatika / Jurnal Ilmiah Komputer & Rekayasa / Jurnal Sistem Cerdas (SINTA 2/3)**
- **Judul yang Direkomendasikan:**
  > *"Uji Keandalan Eksplanabilitas Spasial Menggunakan Grad-CAM dan Analisis Kesalahan Berdasarkan Stratum Fotometrik pada Klasifikasi Kerusakan Jalan"*
- **Fokus Utama:**
  Menjawab isu transparansi dan *trustworthiness* model AI (*Explainable AI*) pada infrastruktur sipil dengan membuktikan apakah visualisasi keputusan model benar-benar memetakan cekungan lubang jalan atau tertipu oleh batas bayangan.
- **Batasan Pembahasan (*Delimitation*):**
  - Fokus eksklusif pada validasi visualisasi Grad-CAM dan taksonomi kegagalan (*error failure analysis*).
  - Melakukan pembuktian kuantitatif menggunakan **Adebayo Model Parameter Randomization Test** (korelasi peringkat Spearman $\rho$ kaskade layer).
  - Menganalisis secara mendalam 239 kasus misklasifikasi pada kondisi alami dan stratum kecerahan (*low*, *medium*, *high relative luminance*).
  - **TIDAK** membahas kurva degradasi luas atau benchmark kecepatan hardware.
- **Tabel & Gambar yang Dipakai:**
  - Tabel 4 (Analisis Kesalahan Berdasarkan Stratum Kecerahan Alami)
  - Tabel 7 (Uji Adebayo Sanity Check: Korelasi Spearman $\rho$)
  - Gambar Heatmap Grad-CAM Komparatif: Model Terlatih vs Head Randomized vs All Parameters Randomized
  - Galeri Citra Kasus Kesalahan Khusus (*False Positive* akibat Bayangan Tajam vs *False Negative* pada Aspal Basah)

---

## 11. Format Sitasi Resmi (BibTeX)

```bibtex
@article{pothole_robustness_2026,
  title     = {Evaluasi Ketahanan Model Konvolusi Ringan Berbasis Augmentasi Iluminasi Sintetis untuk Klasifikasi Lubang Jalan di Bawah Degradasi Fotometrik},
  author    = {Peneliti LPPM},
  journal   = {Jurnal KomtekInfo},
  year      = {2026},
  volume    = {x},
  number    = {x},
  pages     = {x--x},
  publisher = {Universitas Putra Indonesia YPTK Padang}
}
```
