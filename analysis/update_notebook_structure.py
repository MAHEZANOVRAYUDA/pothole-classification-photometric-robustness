import nbformat
from pathlib import Path
import json

root = Path(__file__).resolve().parent.parent
nb_path = root / 'Notebook_Revisi_KomtekInfo_v2_1.ipynb'
nb = nbformat.read(nb_path, as_version=4)

# 1. Update Markdown Header (Cell 0)
academic_header = """# Evaluasi Ketahanan Model Konvolusi Ringan Berbasis Augmentasi Iluminasi Sintetis untuk Klasifikasi Lubang Jalan
**Target Publikasi: Jurnal KomtekInfo (SINTA 3) / RESTI / SinkrOn**  
*Penulis & Peneliti: Tim Riset Computer Vision & Sistem Transportasi Cerdas*  
*Protokol Validasi: Versi 2.1 (Pemisahan Profil Smoke & Research, 5-Fold Stratified Group K-Fold, Paired Bootstrap Bonferroni)*

---

### Abstrak
Inspeksi cacat permukaan jalan, khususnya lubang jalan (*potholes*), merupakan elemen krusial dalam pemeliharaan infrastruktur transportasi dan pencegahan kecelakaan lalu lintas. Meskipun model *Convolutional Neural Networks* (CNN) mencapai akurasi klasifikasi yang tinggi pada citra siang hari beriluminasi optimal, performa model kerap terdegradasi drastis saat berhadapan dengan pergeseran distribusi visual (*covariate shift*) seperti kondisi pencahayaan rendah (*low-light*), bayangan tajam pepohonan/gedung (*harsh shadows*), dan pantulan cahaya (*glare*). Di sisi lain, penerapan pada perangkat komputasi tepi (*edge devices*) menuntut arsitektur yang ringan dan efisien.

Penelitian ini mengevaluasi secara empiris pengaruh augmentasi iluminasi sintetis terhadap ketahanan klasifikasi citra lubang jalan pada dua arsitektur CNN ringan: **EfficientNet-B0** (~5.3M parameter) dan **MobileNetV3-Large** (~5.4M parameter). Evaluasi dilakukan menggunakan protokol validasi silang 5-Fold *Stratified Group K-Fold* (SGKF) pada dataset publik (352 normal, 329 potholes) dengan deduplikasi visual dHash 64-bit untuk mencegah kebocoran data (*data leakage*). Tiga strategi augmentasi dibandingkan secara terkontrol: (1) augmentasi standar (*horizontal flip*), (2) *photometric jitter* (kecerahan dan kontras), serta (3) augmentasi iluminasi sintetis (*low-light* dan *shadow* bertingkat keparahan 1–3). Ketahanan model diuji melintasi 44 kondisi out-of-distribution (1 clean, 13 low-light, 15 shadow, dan 15 operator glare yang tidak dilatih). Efek perlakuan dianalisis menggunakan *Paired Group Bootstrap* (10.000 draw) dengan interval kepercayaan terkoreksi Bonferroni untuk mengendalikan *family-wise error rate*. Selain itu, interpretabilitas model diuji menggunakan Grad-CAM dengan *sanity checks* randomisasi bobot Adebayo, serta diukur profil latensi inferensi batch-1 pada CPU satu *thread* dan GPU. Hasil evaluasi memberikan bukti kuantitatif mengenai kompromi antara ketahanan gangguan pencahayaan ekstrem, performa pada citra bersih, dan efisiensi inferensi perangkat keras.

**Kata Kunci:** *Klasifikasi Lubang Jalan, Ketahanan Pencahayaan, EfficientNet-B0, MobileNetV3-Large, Augmentasi Sintetis, 5-Fold SGKF, Grad-CAM.*

---

### English Abstract
*Road surface distress inspection, particularly pothole classification, is vital for roadway safety and preventative infrastructure maintenance. While Convolutional Neural Networks (CNNs) demonstrate high diagnostic accuracy under canonical daytime conditions, their reliability deteriorates sharply when subjected to real-world illumination shifts, including nocturnal low-light, harsh shadows, and direct glare. Furthermore, mobile and edge inspection platforms necessitate lightweight computational architectures.*

*This study rigorously evaluates the impact of synthetic illumination augmentation on the classification robustness of two lightweight CNN architectures: EfficientNet-B0 (~5.3M parameters) and MobileNetV3-Large (~5.4M parameters). We establish a leakage-free 5-fold Stratified Group K-Fold (SGKF) protocol on an open-access road dataset (352 normal, 329 potholes) reinforced with 64-bit dHash perceptual deduplication. Three training strategies are systematically benchmarked: standard horizontal flip, brightness-contrast photometric jitter, and synthetic illumination augmentation (low-light and shadow, severities 1–3). Robustness is evaluated across 44 out-of-distribution corruption conditions, including untrained glare artifacts. Effect sizes are inferred via paired-group bootstrap resampling (10,000 draws) with Bonferroni-adjusted confidence intervals across 12 comparison families. Model explainability is verified via Grad-CAM paired with Adebayo parameter randomization sanity checks, complemented by batch-1 forward latency profiling on single-threaded CPU and GPU. This research delivers empirical guidelines regarding the trade-offs between illumination robustness, clean-image fidelity, and edge compute efficiency.*

**Keywords:** *Pothole Classification, Illumination Robustness, EfficientNet-B0, MobileNetV3-Large, Synthetic Augmentation, Stratified Group K-Fold, Grad-CAM.*

---

### Rumusan Masalah & Kontribusi Penelitian (Research Questions)
1. **RQ1:** Apakah augmentasi iluminasi sintetis (*low-light* dan *shadow*) meningkatkan Macro-F1 out-of-fold secara signifikan dibandingkan augmentasi standar dan *color jitter* pada EfficientNet-B0 dan MobileNetV3-Large?
2. **RQ2:** Bagaimana konsistensi ketahanan model melintasi tingkat keparahan gangguan (severity 1–5), serta apakah terjadi penurunan performa (*cost of robustness*) pada citra bersih (*clean*) dan operator *glare* yang tidak pernah dilatih?
3. **RQ3:** Bagaimana trade-off antara ketahanan model terhadap biaya komputasi inferensi (latensi P50/P95 batch-1 pada CPU/GPU, FLOPs/MACs, dan memori bobot)?
"""
nb.cells[0].source = academic_header

# 2. Update Config Cell (Tag: 'config')
config_code = """DATA_ROOT=Path('Pothole-dataset')
OUT_ROOT=Path('outputs')
RUN_TAG='protocol_v2_1'
PRETRAINED=True

# Metadata Provenance Dataset Kaggle (Resmi & Terverifikasi)
DATASET_URL = 'https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset'
DATASET_VERSION = 'v1 (atulyakumar98)'
DATASET_LICENSE = 'CC BY-SA 4.0 / Open Database'
SOURCE_GROUPS_CSV = None              # Path('source_groups.csv') bila tersedia

PROTOCOL_VERSION = '2.1.0'
CODE_SIGNATURE = 'dab2dfbbc038e5a938014702dabd969d2e52ec62ab2e163d5261d41353a6cc8a'
CLASS_MAP = {'normal': 0, 'potholes': 1}
CLASS_NAMES = ['normal', 'potholes']
ARCHS = ['efficientnet_b0', 'mobilenet_v3_large']
STRATEGIES = ['standard', 'jitter', 'illum']
KINDS = ['low_light', 'shadow', 'glare']
SEVERITIES = [1, 2, 3, 4, 5]

BASE_CFG = dict(
    img_size=224, batch_size=16, workers=0, amp=True, split_seed=2026,
    n_folds=5, folds=list(range(5)), seeds=[42, 43, 44], tune_seed=17,
    epochs=20, tune_epochs=5, patience=5, lr_grid=[1e-4, 3e-4], weight_decay=1e-3,
    near_hash_distance=4, aug_probability=0.6, aug_max_severity=3, replicas=3,
    bootstrap=10000, bootstrap_chunk=128, alpha=0.05, clean_margin_exploratory=0.02,
    warmup=30, latency_iters=200, cpu_threads=1, cam_samples_per_class=6,
    cam_random_seeds=[2026, 2027, 2028], resume=True, keep_last_completed=False,
    deterministic_warn_only=True
)

SMOKE_OVERRIDES = dict(
    folds=[0], seeds=[42], epochs=2, tune_epochs=1, patience=2,
    lr_grid=[3e-4], replicas=1, bootstrap=200, warmup=3, latency_iters=10,
    cam_samples_per_class=1, cam_random_seeds=[2026]
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
PHOTO_COLS = ['brightness', 'contrast', 'dark_fraction', 'bright_fraction', 'gradient_energy']

def get_profile_config(profile):
    if profile not in {'smoke', 'research'}:
        raise ValueError("Profil harus 'smoke' atau 'research'.")
    config = copy.deepcopy(BASE_CFG)
    if profile == 'smoke':
        config.update(copy.deepcopy(SMOKE_OVERRIDES))
    return config

print('Persiapan konfigurasi selesai. Perangkat:', DEVICE)
print('Pelatihan dimulai pada sel SMOKE atau RESEARCH di bagian bawah.')"""

for c in nb.cells:
    if c.cell_type == 'code' and 'config' in c.metadata.get('tags', []):
        c.source = config_code
        break

# 3. Update Audit Cell (Tag: 'audit_definitions') to guarantee parents=True on audit_staging and safe unlink
for c in nb.cells:
    if c.cell_type == 'code' and 'audit_definitions' in c.metadata.get('tags', []):
        c.source = c.source.replace("audit=OUT/'audit_staging';audit.mkdir(exist_ok=True)",
                                    "audit=OUT/'audit_staging';audit.mkdir(parents=True,exist_ok=True)")
        c.source = c.source.replace("for p in audit.iterdir():\n        if p.is_file():p.unlink()",
                                    "for p in audit.iterdir():\n        if p.is_file():\n            try:p.unlink()\n            except Exception:pass")
        break

# 4. Update GradCAM Cell (Tag: 'gradcam_definitions') for Windows MAX_PATH fix
for c in nb.cells:
    if c.cell_type == 'code' and 'gradcam_definitions' in c.metadata.get('tags', []):
        if "x=x.to(next(self.model.parameters()).device)" not in c.source:
            c.source = c.source.replace("def __call__(self,x,target_class):\n        self.model.eval()\n        with torch.enable_grad():",
                                        "def __call__(self,x,target_class):\n        self.model.eval()\n        x=x.to(next(self.model.parameters()).device)\n        with torch.enable_grad():")
        if "cam_save_path" not in c.source:
            target_str = "np.savez_compressed(cam_dir/f\"{run['arch']}_{run['strategy']}_{row.image_id}.npz\",clean=c['heat'],shadow=s['heat'],target=int(row.label))"
            replacement_str = """cam_file = cam_dir / f"{run['arch']}_{run['strategy']}_{row.image_id}.npz"
                cam_save_path = str(cam_file.resolve())
                if os.name == 'nt' and not cam_save_path.startswith('\\\\\\\\?\\\\'):
                    cam_save_path = '\\\\\\\\?\\\\' + cam_save_path
                np.savez_compressed(cam_save_path, clean=c['heat'], shadow=s['heat'], target=int(row.label))"""
            c.source = c.source.replace(target_str, replacement_str)
        break

nbformat.write(nb, nb_path)
print("Notebook updated successfully with academic header, clean config, and robust path handlers.")
