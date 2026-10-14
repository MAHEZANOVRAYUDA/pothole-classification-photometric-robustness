# Graph Report - Evaluasi Augmentasi Pencahayaan Sintetis terhadap Ketahanan Klasifikasi Lubang Jalan pada EfficientNet-B0 dan MobileNetV3-Large  (2026-10-12)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 105 nodes · 225 edges · 14 communities (9 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- protocol_v2_1.py
- run_experiment
- fit_run
- audit_dataset
- predict_run
- GradCAM
- StreamingNotebookClient
- validate_profile_isolation.py
- validate_protocol.py
- gradcam_analysis
- UnionFind

## God Nodes (most connected - your core abstractions)
1. `run_experiment()` - 20 edges
2. `gradcam_analysis()` - 17 edges
3. `fit_run()` - 15 edges
4. `save_table()` - 10 edges
5. `predict_run()` - 10 edges
6. `split_frames()` - 9 edges
7. `audit_dataset()` - 9 edges
8. `GradCAM` - 8 edges
9. `save_json()` - 8 edges
10. `condition_image()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `audit_dataset()` --calls--> `UnionFind`  [EXTRACTED]
  protocol_v2_1.py → protocol_v2_1.py  _Bridges community 10 → community 3_
- `fit_run()` --calls--> `RoadDataset`  [EXTRACTED]
  protocol_v2_1.py → protocol_v2_1.py  _Bridges community 4 → community 2_
- `gradcam_analysis()` --calls--> `GradCAM`  [EXTRACTED]
  protocol_v2_1.py → protocol_v2_1.py  _Bridges community 5 → community 9_
- `run_experiment()` --calls--> `activate_profile()`  [EXTRACTED]
  protocol_v2_1.py → protocol_v2_1.py  _Bridges community 0 → community 1_
- `prediction_tensor()` --calls--> `condition_id()`  [EXTRACTED]
  protocol_v2_1.py → protocol_v2_1.py  _Bridges community 0 → community 4_

## Import Cycles
- None detected.

## Communities (14 total, 2 thin omitted)

### Community 0 - "protocol_v2_1.py"
Cohesion: 0.24
Nodes (14): activate_profile(), bootstrap_all_scopes(), confusion_counts_by_group(), cpu_name(), environment_info(), f1_from_counts(), freeze_protocol(), get_profile_config() (+6 more)

### Community 1 - "run_experiment"
Cohesion: 0.26
Nodes (15): binary_metrics(), error_analysis(), evaluate_baselines(), evaluate_deep(), finalize_experiment(), metrics_by(), performance_summaries(), plot_confusions() (+7 more)

### Community 2 - "fit_run"
Cohesion: 0.16
Nodes (14): atomic_torch_save(), benchmark_latency(), build_model(), count_conv_linear_macs(), file_sha256(), fit_run(), load_eval_model(), Inisialisasi arsitektur CNN ringan dengan bobot resmi ImageNet-1K V1. Hanya… (+6 more)

### Community 3 - "audit_dataset"
Cohesion: 0.23
Nodes (12): audit_dataset(), calculate_dhash(), condition_image(), condition_seed(), extract_features(), image_stats(), load_rgb(), perturb_image() (+4 more)

### Community 4 - "predict_run"
Cohesion: 0.22
Nodes (8): Dataset, inference_mode, assert_prediction_grid(), condition_id(), make_loader(), predict_run(), RoadDataset, seed_worker()

### Community 6 - "StreamingNotebookClient"
Cohesion: 0.40
Nodes (4): main(), Client yang menyalurkan output stream kernel secara langsung ke stdout terminal., StreamingNotebookClient, NotebookClient

### Community 7 - "validate_profile_isolation.py"
Cohesion: 0.40
Nodes (4): Exception, Check the two-cell workflow without running full research training., stop_before_eda(), StopAfterResearchAudit

### Community 8 - "validate_protocol.py"
Cohesion: 0.40
Nodes (3): interrupt_after_checkpoint(), PlannedInterruption, Exception

### Community 9 - "gradcam_analysis"
Cohesion: 0.33
Nodes (6): gradcam_analysis(), overlay_heatmap(), reset_randomized_model(), safe_spearman(), set_seed(), state_signature()

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GradCAM` connect `GradCAM` to `protocol_v2_1.py`, `gradcam_analysis`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Why does `RoadDataset` connect `predict_run` to `protocol_v2_1.py`, `fit_run`, `audit_dataset`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Why does `UnionFind` connect `UnionFind` to `protocol_v2_1.py`, `audit_dataset`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._