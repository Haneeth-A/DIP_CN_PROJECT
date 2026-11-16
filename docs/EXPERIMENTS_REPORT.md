# WBCBench 2026 — Experiments, Results, and Viva Notes

**Project:** 13-class White Blood Cell (WBC) classification for cancer screening  
**Dataset:** WBCBench 2026 (`Xin-Tian/wbcbench2026`)  
**Courses:** Digital Image Processing (DIP) + Computer Networks (CN)  
**Team:** A. Haneeth, Nivrutha Varshini  
**Guides:** Dr. Mehbub Alam, Dr. Koushiki Roy  
**Repository:** https://github.com/Haneeth-A/DIP_CN_PROJECT

> **Historical document — preserved as-is for provenance.**
> Sections 8, 11, and 12 predate the completed CN work and the final
> `phase2_test` evaluation. For current status use [`../README.md`](../README.md),
> [`CN_RESULTS.md`](CN_RESULTS.md), and [`TEST_REPORT.md`](TEST_REPORT.md).
> The numerical DIP results in Sections 1–7 remain valid.

This document explains what was built, what was run, what the numbers mean, and what is still pending. It is written so that both team members can use it for the report, the PPT, and the viva.

---

## 0. How to use this document

- **For the report:** Sections 1–7 are the DIP/model story. Section 8 is the CN story. Section 9 is deployment. Section 10 is evaluation/reproducibility. Section 11 is limitations. Section 15 lists what each teammate should contribute.
- **For the viva:** Section 13 is a question-and-answer cheat sheet. Section 14 is a glossary of every technical term used.
- **For the code:** Section 3 maps every file to its purpose. Section 16 has exact commands.

---

## 1. Executive summary

We are building a 13-class WBC classifier on WBCBench 2026. The official metric is **macro-F1**, which weights every class equally even though the dataset is extremely imbalanced. The best model so far is:

- **Model:** EfficientNet-B0, ImageNet-pretrained, 224×224 input, 13-class head.
- **Training data:** `phase1_train` + `phase2_train`, both pristine and degraded (58,082 images).
- **Validation data:** `phase2_eval`, both pristine and degraded (10,700 images).
- **Loss:** focal loss (`gamma=2.0`, class-weighted alpha, label smoothing 0.05).
- **Sampler:** square-root inverse-frequency `WeightedRandomSampler`.
- **Batch size / LR / epochs:** 64 / 5e-4 / 10.
- **Best validation results:** combined macro-F1 **0.7144**, pristine **0.7203**, degraded **0.7082**.
- **Previous best (weighted CE):** combined 0.7075, pristine 0.7136, degraded 0.7014.
- **Pristine-only baseline:** pristine 0.7327, degraded 0.6547, pooled ≈0.693.

**Key finding:** mixed-domain training (pristine + degraded) substantially improves degraded-image performance at a small cost to pristine performance. Focal loss gives a further gain, mainly by improving the mid-frequency/hard class **PC** and **PMY**, not the ultra-rare classes **BNE/MMY/VLY**. The ultra-rare classes remain the main bottleneck and are likely affected by visual overlap and label ambiguity, not only by class imbalance.

**What is not done yet:** the CN server/client, the failure demos, the CN performance measurements, the Streamlit model integration, the final `phase2_test` evaluation, and the Hugging Face metadata join (patient-level leakage check).

---

## 2. Problem, dataset, and metric

### 2.1 Task

Given a 368×368 or 368×370 RGB blood-smear image, predict one of 13 WBC classes:

`SNE, LY, MO, EO, BA, VLY, BNE, MMY, MY, PMY, BL, PC, PLY`

The dataset has a **pristine** version and a **degraded** version of the same images. Degradations include noise and blur of different severities. The classifier must work on both clean and degraded inputs.

### 2.2 Dataset counts

| Split | Folder | Images |
|---|---|---:|
| phase1_train | pristine | 8,288 |
| phase2_train | pristine | 24,897 |
| phase2_train | degraded | 24,897 |
| phase2_eval | pristine | 5,350 |
| phase2_eval | degraded | 5,350 |
| phase2_test | pristine | 16,467 |
| phase2_test | degraded | 16,467 |
| **Total** | | **101,716** |

Pristine total: 55,002. Degraded total: 46,714.

Mixed training set used in the main experiments:

- `phase1_train` pristine + `phase2_train` pristine + `phase2_train` degraded = **58,082** images.
- Validation: `phase2_eval` pristine + degraded = **10,700** images.

### 2.3 Class imbalance

Mixed-training class counts:

| Class | Train count | Eval count | Test count |
|---|---:|---:|---:|
| SNE | 26,183 | 2,306 | 8,178 |
| LY | 16,116 | 1,571 | 4,299 |
| MO | 5,649 | 466 | 1,533 |
| EO | 1,777 | 169 | 492 |
| BA | 914 | 60 | 303 |
| VLY | 664 | 106 | 223 |
| BNE | 791 | 71 | 126 |
| MMY | 768 | 47 | 170 |
| MY | 918 | 75 | 204 |
| PMY | 194 | 34 | 40 |
| BL | 3,970 | 412 | 885 |
| PC | 114 | 32 | 12 |
| PLY | 24 | 1 | 2 |

The imbalance ratio SNE:PLY in training is about **1091:1**. This is why plain accuracy is misleading and why **macro-F1** is the official metric. PLY has only 1 evaluation image and 2 test images; PC has 32 evaluation and 12 test images. Their per-class F1 is statistically unstable, and a single mistake can move macro-F1 noticeably.

### 2.4 Why macro-F1

Macro-F1 computes F1 for each class and takes the unweighted mean. A model that ignores rare classes can still have high accuracy, but its macro-F1 will be low. Since the competition uses macro-F1, we optimize and report that. We also report accuracy, balanced accuracy, weighted-F1, per-class precision/recall/F1, and the confusion matrix for diagnosis.

### 2.5 Splits and leakage

We use the **official splits only**. Never random-split the data. Reasons:

- Pristine and degraded versions of the same image are paired. A random split would put one version in train and the other in validation/test, leaking the answer.
- The dataset card notes that some test images share a patient/case with other splits. The exact grouping key is in the Hugging Face metadata (`patient_hash`), which we have not joined yet.
- `phase2_test` is held out. All current numbers are on `phase2_eval`, which is used for model selection. The final model must be evaluated once on `phase2_test`.

---

## 3. Repository and code map

```
Project_DIP/
├── DATA/                          # raw pristine/degraded JPEGs (untouched)
├── DIP/
│   ├── cellpose/
│   │   ├── pilot/                 # 100-image pilot outputs + CSVs
│   │   ├── pilot_complete_results/pilot_100/  # supplied full results snapshot
│   │   ├── source_package/        # supplied 80-image + postprocessing source
│   │   ├── _archives/             # received ZIPs (provenance)
│   │   ├── notebooks/             # Cellpose pilot notebook
│   │   ├── scripts/               # original / reconstructed Cellpose code
│   │   └── docs/                  # pilot README + requirements
│   └── training/
│       ├── app.py                 # Streamlit UI (currently upload/preview only)
│       ├── configs/base.yaml      # default training config
│       ├── data/manifests/
│       │   └── quality_features.csv  # 101,716-row combined manifest
│       ├── checkpoints/<run>/best.pt # best validation checkpoint
│       ├── results/<run>/
│       │   ├── history.csv        # per-epoch metrics
│       │   ├── best_metrics.json  # best validation metrics
│       │   ├── config.yaml        # exact run config
│       │   └── eval_{pristine,degraded}/  # predictions, metrics, confusion matrix
│       ├── scripts/
│       │   ├── build_manifests.py # handover CSVs -> one manifest
│       │   ├── download_metadata.py  # HF metadata download (needs auth)
│       │   ├── validate_features.py  # parity check vs Nivrutha's CSVs
│       │   ├── train.py           # training CLI
│       │   ├── evaluate.py        # checkpoint evaluation
│       │   ├── ensemble_eval.py   # ensemble / TTA evaluation
│       │   └── finetune_rare.py   # rare-class synthetic augmentation
│       ├── src/
│       │   ├── data/
│       │   │   ├── labels.py      # fixed 13-class order
│       │   │   ├── manifest.py    # build/load/join manifest
│       │   │   └── dataset.py     # WBCDataset + transforms
│       │   ├── models/efficientnet.py  # model factory
│       │   ├── quality/features.py     # 7 hand-crafted features
│       │   └── training/
│       │       ├── losses.py      # class weights, sampler weights, focal loss
│       │       └── metrics.py     # accuracy, macro-F1, per-class, confusion
│       └── tests/test_features.py # 9 passing feature tests
├── CN/
│   ├── src/cn/protocol.py         # TCP framing + SHA-256 protocol
│   └── tests/                     # transport, backend, policy, integration
├── model_share/deploy/            # quantized ONNX deployment bundle
├── feature_analysis/              # quality features and analysis
└── docs/                          # consolidated technical documentation
```

### 3.1 What each important file does

- `src/data/labels.py`: fixed `LABELS` order and `CLASS_TO_IDX` / `IDX_TO_CLASS`. The order must never change, because checkpoints store label indices.
- `src/data/manifest.py`: combines the seven quality-feature CSVs into one table, adds `split` and `folder`, normalizes `image_id` to 8 digits, and optionally joins HF metadata.
- `src/data/dataset.py`: filters the manifest by split/folder/severity, loads `{data_root}/{folder}/{split}/{image_id}.jpg`, applies transforms, returns image + label + metadata.
- `src/quality/features.py`: the seven verified image-quality features.
- `src/models/efficientnet.py`: builds `efficientnet_b0`, `efficientnet_v2_s`, `resnet50`, or `swin_t` with ImageNet weights and a new 13-class head.
- `src/training/losses.py`: class weights (`inv`, `sqrt`, `effective`), per-sample sampler weights, focal loss, and the criterion factory.
- `src/training/metrics.py`: all metrics and the confusion matrix.
- `scripts/train.py`: end-to-end training with AMP, AdamW, cosine annealing, class weights, sampler, best-checkpoint saving.
- `scripts/evaluate.py`: loads `best.pt`, runs inference, writes predictions/metrics/confusion matrix.
- `src/cn/protocol.py`: the application-layer frame format and message types for the CN component.

---

## 4. Data pipeline and quality features

### 4.1 Manifest

`data/manifests/quality_features.csv` has 101,716 rows and columns:

`image_id, split, folder, class_label, severity_label, laplacian_variance, median_filter_residual, brightness, contrast, extreme_pixel_percentage, rgb_colour_variation, edge_density`

It is built from Nivrutha's seven quality-feature CSVs (the handover folder also contains `WBC_Processing_Manifest.csv` and `phase2_train_laplacian_investigation.csv`). The `folder` column (`pristine`/`degraded`) is parsed from the filename; the `split` column is inserted from the filename.

### 4.2 The seven features

| Feature | Formula | What it captures |
|---|---|---|
| `laplacian_variance` | `var(Laplacian(gray, CV_64F))` | Focus/sharpness. High = sharp or noisy; low = blurry. |
| `median_filter_residual` | `mean(abs(gray - medianBlur(gray, 3)))` | High-frequency noise. High = salt-and-pepper/noise. |
| `brightness` | `mean(gray)` | Overall illumination. |
| `contrast` | `std(gray)` | Dynamic range. |
| `extreme_pixel_percentage` | `100 * mean((gray <= 5) or (gray >= 250))` | Clipping/saturation. |
| `rgb_colour_variation` | `mean(std(R), std(G), std(B))` | Colour richness. |
| `edge_density` | `100 * count(Canny(gray, 100, 200)) / pixels` | Structural edges. |

### 4.3 Validation of the port

`scripts/validate_features.py --n 50` checks our implementation against Nivrutha's CSVs:

- Laplacian variance, median-filter residual, brightness, contrast, edge density: exact (tolerance `1e-6`).
- Extreme-pixel percentage: exact. Examples: `01416766` = 3.181551, `01424428` = 5.431829.
- RGB colour variation: matches to <0.01; tolerance set to 0.05 because of JPEG decode rounding.

`tests/test_features.py` has 9 passing tests. This matters because the same feature code is used by the CN server and the Streamlit app, so the port must be trustworthy.

### 4.4 The degradation-type insight

Severity is **not** a single ordered scale. The degraded set is a mixture of types:

- **Mild degradation is closer to salt-and-pepper noise.** Noise adds high-frequency spikes, so Laplacian variance is very high (observed around 127k) and the median-filter residual is high.
- **Extreme degradation is closer to Gaussian blur.** Blur removes high frequencies, so Laplacian variance collapses (observed around 1.5) and the median-filter residual is low.

Therefore, severity alone is misleading. The right routing is **type-aware**:

- Noise → denoise (median filter or non-local means) → classifier.
- Blur → skip or deblur → classifier.
- Clean → direct classifier.

This is a DIP insight, not just a preprocessing trick. It also explains why a single global preprocessing step would hurt some images while helping others.

### 4.5 Hugging Face metadata (pending)

`scripts/download_metadata.py` downloads `metadata/labels.csv` and `metadata/degradation_params.csv`. These provide:

- `original_image_id` for pairing pristine/degraded predictions.
- `patient_hash` for patient-disjoint splits and leakage checks.
- Exact degradation operations and parameters.

This requires an authenticated Hugging Face account with access to the dataset (`hf auth login` or `HF_TOKEN`). It has not been run yet. Until it is, patient-level leakage is unverified.

---

## 5. Model and training setup

### 5.1 Model

- **EfficientNet-B0**, ImageNet-pretrained (`IMAGENET1K_V1`).
- The original classifier is replaced with `Dropout(0.2) + Linear(1280, 13)`.
- Why B0: small, fast, strong transfer learning, fits an 8 GB laptop GPU. Architecture is not the bottleneck at this data scale; data quality and class imbalance matter more.
- Other supported models for controlled comparison: `efficientnet_v2_s`, `resnet50`, `swin_t`.

### 5.2 Transforms

Training:

- `Resize((img_size, img_size))`
- `RandomHorizontalFlip(0.5)`
- `RandomVerticalFlip(0.5)`
- `RandomRotation(15°)`
- `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)`
- `ToTensor()`
- `Normalize(ImageNet mean/std)`

Validation/test:

- `Resize((img_size, img_size))`
- `ToTensor()`
- `Normalize(ImageNet mean/std)`

The flips and rotations are safe for cell images because class identity does not depend on orientation. Colour jitter is mild to avoid changing stain appearance too much.

### 5.3 Loss

Default: **weighted cross-entropy** with class weights and label smoothing 0.05.

- `weighted_ce`: `CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`.
- `focal`: `FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.05)`.
- `ce`: plain cross-entropy (used for ablation).

**Label smoothing** prevents the model from becoming overconfident and improves calibration.

**Focal loss** is `-(1 - p_t)^gamma * log(p_t)`. Easy examples (high `p_t`) are down-weighted; hard examples stay important. `alpha` adds class weighting. The focal-loss value is not comparable to cross-entropy because of the `(1-p_t)^gamma` factor; compare F1, not loss, across losses.

### 5.4 Class imbalance handling

We use two complementary mechanisms:

1. **WeightedRandomSampler** with per-sample weights from `compute_class_weights`.
2. **Class-weighted loss** with the same weights (for `weighted_ce` or as focal `alpha`).

Weight modes:

- `inv`: full inverse frequency, `N / (C * count)`. Very aggressive.
- `sqrt`: `1 / sqrt(count)`, normalized to mean 1. Gentler.
- `effective`: Cui et al. effective number, `(1 - beta) / (1 - beta^count)`, `beta=0.999`, normalized. Middle ground.

For the mixed-training set, relative to SNE:

| Mode | PLY | PC | PMY | BNE | MMY | VLY |
|---|---:|---:|---:|---:|---:|---:|
| sqrt | 33.0× | 15.2× | 11.6× | 5.8× | 5.8× | 6.3× |
| effective | 42.2× | 9.3× | 5.7× | 1.8× | 1.9× | 2.1× |
| inv | 1091× | 230× | 135× | 33× | 34× | 39× |

We deliberately avoid `inv` for the main runs: repeating PLY ~1091× would overfit 24 training images. We also do **not** literally duplicate files.

### 5.5 Optimizer, scheduler, AMP

- **AdamW**: Adam with decoupled weight decay (`weight_decay=1e-4`).
- **CosineAnnealingLR**: `T_max=epochs`, `eta_min=lr * 0.01`; the learning rate decays smoothly.
- **AMP** (automatic mixed precision): `torch.amp.autocast` + `GradScaler`; faster and lower memory on the RTX 4060.
- **Gradient clipping**: `max_norm=5.0` for stability.
- **Seed**: 42 by default; seeds Python, NumPy, PyTorch, CUDA.

### 5.6 Checkpoints

- `last.pt` saved every epoch.
- `best.pt` saved when validation macro-F1 improves.
- Contents: `model_state`, `optimizer_state`, `epoch`, `best_macro_f1`, `config`, `labels`.
- Not saved: scheduler state, scaler state, RNG state. There is **no `--resume`** flag. Continuing training exactly is not supported without a code change.

### 5.7 Evaluation

`compute_metrics` returns:

- accuracy
- balanced accuracy (mean per-class recall)
- macro-F1 (official)
- weighted-F1
- per-class precision, recall, F1, support
- confusion matrix

`scripts/evaluate.py` writes `predictions.csv`, `metrics.json`, `per_class_metrics.csv`, and `confusion_matrix.png`.

---

## 6. Experiments and results

All runs use EfficientNet-B0, ImageNet pretrained, seed 42. “Combined” means macro-F1 on the pooled `phase2_eval` pristine + degraded set (10,700 images). “Pristine” and “Degraded” are separate 5,350-image evaluations.

### 6.1 Baseline and resolution/batch runs

| Run | Train data | Val data | Img / batch / LR | Combined | Pristine | Degraded |
|---|---|---|---:|---:|---:|---:|
| `effb0_pristine_v1` | phase1+phase2 pristine | eval pristine | 224 / 32 / 3e-4 | ≈0.693 (pooled) | 0.7327 | 0.6547 |
| `effb0_288_bs64_pristine` | phase1+phase2 pristine | eval pristine+degraded | 288 / 64 / 5e-4 | 0.6916 | 0.7261 | 0.6577 |

### 6.2 Mixed-domain 2×2 factorial

Train: phase1+phase2 pristine + phase2 degraded (58,082). Val: eval pristine+degraded (10,700).

| Run | Img / batch / LR | Combined | Pristine | Degraded |
|---|---|---|---:|---:|---:|
| `effb0_224_bs32_mixed` | 224 / 32 / 3e-4 | 0.6927 | 0.7029 | 0.6830 |
| `effb0_224_bs64_mixed` | 224 / 64 / 5e-4 | 0.7075 | 0.7136 | 0.7014 |
| `effb0_288_bs32_mixed` | 288 / 32 / 3e-4 | 0.6961 | 0.7098 | 0.6824 |
| `effb0_288_bs64_mixed` | 288 / 64 / 5e-4 | 0.7011 | 0.7106 | 0.6914 |

### 6.3 Loss, sampler, and schedule ablations

All at 224 / 64 / 5e-4, mixed data.

| Run | Epochs | Loss | Sampler | Combined | Pristine | Degraded |
|---|---:|---|---|---:|---:|---:|
| `effb0_224_bs64_mixed` | 15 | weighted CE | sqrt | 0.7075 | 0.7136 | 0.7014 |
| `effb0_224_bs64_mixed_e10` | 10 | weighted CE | sqrt | 0.6995 | 0.7075 | 0.6916 |
| `effb0_224_bs64_mixed_focal` | 10 | focal | sqrt | **0.7144** | **0.7203** | **0.7082** |
| `effb0_224_bs64_mixed_eff` | 10 | weighted CE | effective | 0.7073 | 0.7151 | 0.6991 |

### 6.4 Best model: per-class results

Focal loss vs the previous best weighted-CE model:

**Pristine (F1):**

| Class | Weighted CE | Focal | Change |
|---|---:|---:|---:|
| SNE | 0.974 | 0.976 | +0.002 |
| LY | 0.961 | 0.959 | -0.002 |
| MO | 0.921 | 0.911 | -0.010 |
| EO | 0.983 | 0.991 | +0.008 |
| BA | 0.992 | 0.992 | 0.000 |
| VLY | 0.601 | 0.577 | -0.024 |
| BNE | 0.331 | 0.320 | -0.011 |
| MMY | 0.495 | 0.485 | -0.010 |
| MY | 0.680 | 0.671 | -0.009 |
| PMY | 0.609 | 0.657 | +0.048 |
| BL | 0.905 | 0.904 | -0.001 |
| PC | 0.825 | 0.921 | +0.096 |
| PLY | 0.000 | 0.000 | 0.000 |

**Degraded (F1):**

| Class | Weighted CE | Focal | Change |
|---|---:|---:|---:|
| SNE | 0.969 | 0.972 | +0.003 |
| LY | 0.958 | 0.956 | -0.002 |
| MO | 0.913 | 0.906 | -0.007 |
| EO | 0.929 | 0.944 | +0.015 |
| BA | 0.949 | 0.974 | +0.025 |
| VLY | 0.621 | 0.598 | -0.023 |
| BNE | 0.333 | 0.310 | -0.023 |
| MMY | 0.500 | 0.500 | 0.000 |
| MY | 0.653 | 0.644 | -0.009 |
| PMY | 0.615 | 0.645 | +0.030 |
| BL | 0.891 | 0.896 | +0.005 |
| PC | 0.787 | 0.862 | +0.075 |
| PLY | 0.000 | 0.000 | 0.000 |

Focal’s overall gain comes mainly from **PC** and **PMY**, plus small gains on EO/BA. The weakest classes **BNE/MMY/VLY** do not improve. This is the central analysis point.

### 6.5 Best-model summary

| Metric | Pristine | Degraded |
|---|---:|---:|
| Accuracy | 0.9325 | 0.9264 |
| Balanced accuracy | 0.7202 | 0.7027 |
| Macro-F1 | 0.7203 | 0.7082 |
| Weighted-F1 | 0.9330 | 0.9270 |

---

## 7. What the results mean

### 7.1 Mixed-domain training helps degraded images

The pristine-only baseline drops from 0.7327 (pristine) to 0.6547 (degraded). Mixed training raises degraded to 0.7014 (weighted CE) and 0.7082 (focal), while pristine stays around 0.714–0.720. The model learns features that survive both domains. The cost to pristine is small compared with the gain on degraded. Since the final evaluation includes both, the net effect is positive.

### 7.2 Focal loss helps, but not where we expected

Focal loss down-weights easy examples. The biggest gain is **PC** (0.825 → 0.921 pristine; 0.787 → 0.862 degraded) and **PMY** (0.609 → 0.657; 0.615 → 0.645). PC is rare (114 training images) but apparently visually separable once the model focuses on it. PMY also benefits.

The ultra-rare classes **BNE, MMY, VLY, MY** do not improve. The confusion analysis shows why:

- SNE → BNE and BNE → SNE are the largest confusions.
- VLY → LY and LY → VLY are large.
- BL → LY is large.

These are visually overlapping classes, and some labels may be ambiguous. More loss weighting alone cannot fix label ambiguity or insufficient evidence. The next step should be class-specific handling (targeted oversampling, two-stage fine-tuning, or label-quality checks), not just a stronger loss.

### 7.3 Effective sampler: accuracy up, macro-F1 down

The effective-number sampler gives strong weights to PLY (42×), PC (9.3×), and PMY (5.7×). It improves **BNE** (0.350 pristine / 0.374 degraded) and achieves the highest accuracy (0.9389 pristine / 0.9327 degraded), but it hurts MMY/VLY/MY, so its macro-F1 is lower than focal (0.7151 / 0.6991 vs 0.7203 / 0.7082). This is a good example of why accuracy is misleading under imbalance: the model gets more common-class predictions right while losing rare-class balance.

### 7.4 Resolution: 288 does not give a consistent gain

- At batch 32: 288/32 combined 0.6961 vs 224/32 0.6927. The difference is tiny (+0.003) and within rare-class noise.
- At batch 64: 288/64 combined 0.7011 vs 224/64 0.7075. 288 is worse.
- 288 also costs more compute and heat: ~186 s/epoch vs ~110 s/epoch at 224.

Conclusion: 224 is the better operating point for this dataset and hardware. EfficientNet-B0 at 224 is sufficient.

### 7.5 Batch size: 224/64 wins, but the comparison is confounded with LR

The 224/64 mixed run used LR 5e-4, while 224/32 used LR 3e-4. So the 224/64 gain (0.7075 vs 0.6927) may come from the larger batch, the higher LR, or both. A clean batch-size ablation would run 224/32 at 5e-4 or 224/64 at 3e-4. This is a limitation to state honestly. The resolution comparison is clean because both batch-32 runs used 3e-4 and both batch-64 runs used 5e-4.

### 7.6 Early best epochs and overfitting

Best epochs:

- 224/64 mixed: epoch 9 of 15.
- 288/32 mixed: epoch 7 of 15.
- 224/64 focal: epoch 8 of 10.
- 224/64 effective: epoch 9 of 10.

The models overfit after roughly 8–9 epochs. `best.pt` protects the final result, but a shorter schedule or early stopping could save compute. The 10-epoch weighted-CE run did **not** improve (0.6995 vs 0.7075), so simply shortening the cosine schedule is not a win. The best checkpoint, not the last epoch, is what we report.

### 7.7 PLY and PC are statistically unstable

PLY has 1 evaluation image and 2 test images. PC has 32 evaluation and 12 test images. Macro-F1 includes PLY with F1 0.0, which drags the average; one PLY prediction changes the metric a lot. Report these per-class numbers with support counts and explicitly call them unstable. Do not over-interpret small differences in overall macro-F1 that may be driven by PLY/PC.

### 7.8 Thermal and compute notes

The RTX 4060 Laptop (8 GB) runs at 84–87 °C with the fan at maximum and shows `SW Thermal Slowdown` active at times. This is the laptop limit. Do not overclock. 288/64 runs are hotter and slower. 224/64 is the practical choice.

---

## 8. Computer Networks component

> **Outdated:** The CN transport, edge–fog policy, 20/20 real-TCP validation,
> tests, demo, reports, and slides are now complete under `CN/`. This section
> describes the original plan, not the final implementation.

### 8.1 Why a custom TCP protocol

The CN course is about sockets, framing, error control, ARQ, concurrency, and performance. A plain HTTP REST call would hide all of that. A custom application-layer protocol over TCP demonstrates the course concepts directly and is easy to defend in a viva.

### 8.2 Architecture

```
CLIENT                                      SERVER (threaded)
  read image                                  accept connection
  compute whole-file SHA-256                  parse handshake
  split into frames                           receive/verify frames
  send frame, wait ACK, retransmit            reassemble + verify whole-file SHA-256
  receive JSON result                         run DIP + ML pipeline
  display result                              return JSON result
```

The server pipeline:

`decode image -> 7 quality features -> degradation type/severity -> optional preprocessing -> EfficientNet-B0 -> JSON`

The JSON response should include:

`predicted_class, confidence, top3, quality_features, degradation_type, severity, preprocessing_applied, server_latency_ms, bytes_received, integrity_ok`

### 8.3 Implemented protocol (`src/cn/protocol.py`)

Actual wire format, big-endian:

```
magic(4s) | version(B) | msg_type(B) | seq(I) | total(I) | payload_len(I)
payload(payload_len)
sha256(payload)(32s)
```

- `magic = b"WBCN"`, `version = 1`.
- Header size: 18 bytes.
- `MAX_PAYLOAD = 16 MiB`.
- Per-frame SHA-256 (32 bytes) plus a planned whole-file SHA-256.
- Message types: `HELLO=1`, `HELLO_ACK=2`, `FILE_META=3`, `META_ACK=4`, `FILE_CHUNK=5`, `CHUNK_ACK=6`, `NACK=7`, `DONE=8`, `DONE_ACK=9`, `ERROR=10`.
- Helpers: `pack_message`, `send_message`, `send_json`, `recv_exact`, `recv_message`, `sha256_file`.

This module is implemented. The server, client, and failure demos are **not yet implemented**; they are the next CN task.

### 8.4 Planned reliability and flow control

- **Stop-and-wait ARQ:** send one frame, wait for `CHUNK_ACK`, timeout and retransmit on loss.
- **NACK on checksum failure:** retransmit the corrupted frame.
- **Whole-file SHA-256:** verify reassembly.
- **Optional sliding window:** window size W with cumulative ACK, to compare throughput against stop-and-wait.
- **Thread-per-client server:** multiple clients concurrently.

### 8.5 Demo scenarios for the viva

1. Normal transfer: image sent, ACKed, classified, JSON displayed.
2. Lost frame: drop a frame; timeout; retransmit; transfer completes.
3. Corrupted frame: flip a byte; checksum fails; NACK; retransmit.
4. Wrong file hash: server reports integrity failure.
5. Concurrent clients: two clients upload at once; both get correct results.
6. Chunk-size sweep: throughput/latency vs 1 KB, 4 KB, 16 KB, 64 KB.
7. Optional window sweep: stop-and-wait vs window W.

### 8.6 Measurements for the report

- End-to-end latency (connect → result).
- Throughput (KB/s) vs chunk size.
- Number of retransmissions and total frames.
- RTT per frame.
- Server inference time vs network time.
- Optional TCP vs UDP comparison for the same image.
- Optional Wireshark capture of the TCP handshake and segments.

### 8.7 Course mapping

| Component | CN concept | Course link |
|---|---|---|
| TCP connect / port | Client–server, transport layer | Lab 1 |
| Frame header + length | Framing, data-link control | Ch. 11 / Lab 2 |
| Sequence numbers | Ordering, ARQ | Ch. 11 / Labs 4–5 |
| ACK / NACK / timeout | Error and flow control | Lab 4 |
| Retransmission | Stop-and-wait ARQ | Lab 4 |
| Optional window | Sliding window / Go-Back-N | Lab 5 |
| Per-frame SHA-256 + file SHA-256 | Error detection + integrity | Lab 2 / reference |
| Threaded server | Concurrent clients | Lab 1 |
| Latency/throughput vs chunk size | Network performance | Course mechanics |

---

## 9. Deployment

`app.py` is a Streamlit app that currently supports upload, preview, and image metadata. It does **not** run the model yet. The planned integration:

- Upload image.
- Show the 7 quality features and a degradation type/severity estimate.
- Run the same `predict()` function used by the CN server.
- Show predicted class, confidence, top-3, and optional preprocessing/segmentation overlay.

Using the same `predict()` function avoids duplicated logic between the app and the server.

---

## 10. Evaluation and reproducibility

### 10.1 Evaluation protocol

- Model selection on `phase2_eval` (pristine + degraded).
- Final evaluation on `phase2_test` (pristine + degraded) once, after the model is frozen.
- Report macro-F1 as the primary metric; include balanced accuracy, accuracy, weighted-F1, per-class metrics, and confusion matrices.
- Report support counts for every per-class number.
- Call out PLY/PC instability.
- Do not tune on `phase2_test`.

### 10.2 Environment

- Python 3.10.22 in `.venv`.
- torch 2.6.0+cu124, torchvision 0.21.0+cu124.
- CUDA on RTX 4060 Laptop, 8 GB.
- OpenCV 5.0.0, scikit-learn 1.7.2, NumPy 2.2.6, Streamlit 1.65.0, pytest.
- No `.venv/bin/pip`; install with `uv pip install --python .venv/bin/python`.

### 10.3 Reproducibility

Each run stores `config.yaml`, `history.csv`, `best_metrics.json`, and the best/last checkpoints. Seeds are fixed. The exact commands are in Section 16.

---

## 11. Limitations and risks

> **Outdated items:** Item 2 is done (`phase2_test` evaluated; see
> `../results/test_final/`). Item 7 is closed: Cellpose was tested and rejected,
> and it is not integrated or planned future work. Item 8 is done (CN
> transport/policy/demos implemented). Item 9 remains out of scope for this
> handover (Streamlit integration is a separate application task). Items 1, 3,
> 4, 5, and 10 still apply.

1. **Patient-level leakage is unverified.** HF metadata is not downloaded. Use `patient_hash` to check splits before claiming generalization.
2. **Final test set not evaluated.** All current numbers are on `phase2_eval`. Run `phase2_test` once at the end.
3. **Batch-size/LR confound.** 224/64 used LR 5e-4; 224/32 used 3e-4. A clean batch ablation is pending.
4. **Rare-class bottleneck.** BNE/MMY/VLY/MY remain weak; likely visual overlap/label ambiguity, not only imbalance.
5. **PLY/PC instability.** PLY has 1 eval / 2 test images; PC has 32 eval / 12 test images. Macro-F1 is unstable because of this.
6. **No GAN.** Deliberate decision: time/risk; augmentation and loss rebalancing are the chosen tools.
7. **Cellpose experiment rejected.** Cellpose crops were tested and did not beat the full-image model; Cellpose is not integrated and is not future work.
8. **CN server/client and demos are not implemented yet.** Only `protocol.py` exists.
9. **Streamlit app is not integrated with the model yet.**
10. **No `--resume`.** Checkpoints do not save scheduler/scaler/RNG state; exact continuation needs code changes.
11. **Thermal throttling.** 84–87 °C on the laptop; 288/64 is hot and slow. Do not overclock.
12. **Do not commit `DATA/`.** The licence forbids redistribution.

---

## 12. Next steps (prioritized)

> **Outdated:** Items 1, 6, 8, 9, and 10 are complete. Item 11 (seed variance),
> item 3 (clean batch/LR ablation), item 4 (rare-class rescue), and item 12
> (report/paper/PPT) remain open. Streamlit integration (item 10) is out of
> scope for this handover.

1. **Evaluate the best focal model on `phase2_test`** (pristine and degraded). This is the held-out official test set.
2. **Download HF metadata and rebuild the manifest.** Verify patient-disjoint splits and join `original_image_id` for paired analysis.
3. **Run a clean batch/LR ablation:** 224/32 at LR 5e-4 or 224/64 at LR 3e-4.
4. **Try focal + effective combined**, and/or class-specific oversampling for BNE/MMY/VLY.
5. **Two-stage fine-tuning:** train mixed, then fine-tune the head/backbone with a balanced sampler and low LR.
6. **TTA and ensemble:** horizontal flip/rotations, and ensemble focal + effective + 288/64.
7. **Degradation-type-aware routing:** noise → denoise, blur → skip/deblur; compare with and without.
8. **Optional model comparison:** EfficientNetV2-S or Swin-T at 224/32. Architecture is likely not the bottleneck, so keep it as a controlled comparison.
9. **Implement the CN server/client, failure demos, and measurements.**
10. **Integrate Streamlit with `predict()`.**
11. **Seed variance:** run the best config with 2–3 seeds and report mean ± std.
12. **Write the report/PPT** using Section 15.

---

## 13. Viva Q&A cheat sheet

**Q1. What is the task?**  
13-class WBC classification on WBCBench 2026, evaluated by macro-F1 on pristine and degraded images.

**Q2. Why macro-F1?**  
It weights all classes equally, so rare classes matter. Accuracy would be dominated by SNE/LY.

**Q3. Why EfficientNet-B0?**  
Small, pretrained, strong transfer learning, fits an 8 GB GPU. Architecture is not the bottleneck; data and imbalance are.

**Q4. Why not a GAN?**  
Time and risk. Augmentation, mixed-domain training, and loss rebalancing gave gains without GAN instability.

**Q5. What is transfer learning?**  
Start from ImageNet weights, replace the 13-class head, fine-tune on WBC images.

**Q6. What is the classifier head?**  
`Dropout(0.2) + Linear(1280, 13)` after the EfficientNet feature extractor.

**Q7. What are the seven quality features?**  
Laplacian variance, median-filter residual, brightness, contrast, extreme-pixel percentage, RGB colour variation, edge density. They capture sharpness, noise, illumination, contrast, clipping, colour, and edges.

**Q8. How did you validate the feature port?**  
Against Nivrutha’s CSVs: exact for five features and extreme-pixel percentage; RGB variation within 0.01 (tolerance 0.05) due to JPEG rounding. Nine tests pass.

**Q9. What is the degradation-type insight?**  
Severity is a mixture. Mild is noise-like (high Laplacian variance, high median residual); extreme is blur-like (low Laplacian variance, low median residual). Route by type, not severity alone.

**Q10. What is mixed-domain training?**  
Train on pristine and degraded versions together. It improves degraded performance at a small pristine cost.

**Q11. What is WeightedRandomSampler?**  
A sampler that draws rare classes more often using per-sample weights, with replacement.

**Q12. What is sqrt weighting?**  
`w = 1 / sqrt(count)`, normalized to mean 1. It is gentler than full inverse frequency.

**Q13. What is effective-number weighting?**  
Cui et al.: `(1 - beta) / (1 - beta^count)`, `beta=0.999`, normalized. It is a middle ground between sqrt and inverse.

**Q14. Why not full inverse frequency?**  
It gives PLY ~1091× the weight of SNE, which overfits 24 training images.

**Q15. What is focal loss?**  
`-(1 - p_t)^gamma * log(p_t)`. It down-weights easy examples and focuses on hard ones. We use `gamma=2` and class-weighted alpha.

**Q16. Did focal loss fix the rare classes?**  
No. It improved PC and PMY, but BNE/MMY/VLY/MY stayed weak. Those classes are visually overlapping/ambiguous, not just underweighted.

**Q17. What is label smoothing?**  
A small amount (0.05) added to the target distribution to prevent overconfidence.

**Q18. What is AdamW?**  
Adam with decoupled weight decay; adaptive learning rates plus regularization.

**Q19. What is cosine annealing?**  
The learning rate follows a cosine curve down to 1% of the base LR over the run.

**Q20. What is AMP?**  
Automatic mixed precision: FP16 where safe, FP32 where needed, with a GradScaler. Faster and lower memory.

**Q21. Why 224 instead of 288?**  
288 gave no consistent gain and was hotter/slower. 224/64 was the best operating point.

**Q22. Why batch 64?**  
Better GPU utilization and stable gradients; but the comparison is confounded with LR (5e-4 vs 3e-4), so a clean ablation is pending.

**Q23. Why is the best epoch early?**  
The model overfits after ~8–9 epochs. We save `best.pt` by validation macro-F1, so the final model is the best checkpoint, not the last epoch.

**Q24. What is balanced accuracy?**  
The mean per-class recall. It is less sensitive to class imbalance than plain accuracy.

**Q25. What does the confusion matrix show?**  
SNE↔BNE, VLY↔LY, and BL→LY are the main confusions. This supports the label-overlap/ambiguity explanation.

**Q26. Why are PLY and PC problematic?**  
PLY has 1 eval / 2 test images; PC has 32 eval / 12 test images. Their F1 is unstable and can swing macro-F1.

**Q27. What is data leakage here?**  
Pristine/degraded pairs of the same image, and possibly the same patient, appearing in different splits. We use official splits; metadata is needed to verify patient disjointness.

**Q28. What is the CN protocol?**  
A custom TCP application-layer protocol with framing, sequence numbers, per-frame SHA-256, ACK/NACK, stop-and-wait retransmission, and an optional sliding window.

**Q29. Why TCP and not UDP?**  
TCP gives reliable, ordered delivery. We still implement application-layer ARQ to demonstrate the course concepts; a UDP comparison is optional.

**Q30. What is framing?**  
A header with a length field so the receiver knows where each message ends.

**Q31. What is stop-and-wait ARQ?**  
Send one frame, wait for ACK; on timeout, retransmit. Simple but low throughput.

**Q32. What is sliding window?**  
Multiple frames in flight with cumulative ACKs; higher throughput than stop-and-wait.

**Q33. What is thread-per-client?**  
The server spawns a thread per connection so multiple clients can be served concurrently.

**Q34. What CN measurements will you report?**  
Latency, throughput vs chunk size, retransmissions, RTT, inference vs network time, optional TCP vs UDP and Wireshark.

**Q35. What is the deployment?**  
A Streamlit app for upload, quality report, prediction, confidence, top-3, and optional preprocessing/segmentation.

**Q36. What are the main limitations?**  
Rare-class bottleneck, unverified patient leakage, no final test evaluation yet, batch/LR confound, CN/Streamlit incomplete, thermal limits.

**Q37. What would you do with more time?**  
Metadata/leakage check, final test evaluation, clean batch/LR ablation, focal+effective, two-stage fine-tuning, TTA/ensemble, type-aware routing, CN measurements, seed variance.

---

## 14. Glossary

- **Macro-F1:** unweighted mean of per-class F1. Treats all classes equally.
- **Weighted-F1:** mean of per-class F1 weighted by support. Dominated by common classes.
- **Balanced accuracy:** mean per-class recall.
- **Support:** number of true examples for a class.
- **Confusion matrix:** rows = true class, columns = predicted class.
- **Transfer learning:** reuse ImageNet features and fine-tune on the target dataset.
- **EfficientNet-B0:** a compact CNN with compound scaling; strong accuracy/compute trade-off.
- **Logits:** raw model outputs before softmax.
- **Cross-entropy:** standard classification loss.
- **Weighted CE:** cross-entropy with per-class weights.
- **Focal loss:** down-weights easy examples with `(1-p_t)^gamma`.
- **Label smoothing:** softens targets to reduce overconfidence.
- **AdamW:** Adam with decoupled weight decay.
- **Cosine annealing:** smooth LR decay following a cosine curve.
- **AMP:** automatic mixed precision (FP16/FP32).
- **GradScaler:** scales loss to prevent FP16 underflow.
- **Gradient clipping:** caps gradient norm for stability.
- **WeightedRandomSampler:** samples examples with given probabilities.
- **Effective number:** Cui et al. class-balanced weighting.
- **Inverse frequency:** weight proportional to `1/count`.
- **Sqrt weighting:** weight proportional to `1/sqrt(count)`.
- **Overfitting:** training performance improves while validation worsens.
- **Best checkpoint:** the epoch with the highest validation macro-F1.
- **Pristine/degraded:** clean vs artificially degraded versions of the same image.
- **Severity:** the degradation label (mild/moderate/extreme), which is a mixture of types.
- **Laplacian variance:** variance of the Laplacian; high for sharp/noisy, low for blurry.
- **Median-filter residual:** mean absolute difference from a 3×3 median filter; high for noise.
- **Canny edge density:** fraction of edge pixels.
- **Framing:** delimiting messages with a header and length.
- **ARQ:** automatic repeat request; retransmission on loss/error.
- **Stop-and-wait:** one frame in flight, wait for ACK.
- **Sliding window:** multiple frames in flight, cumulative ACK.
- **SHA-256:** cryptographic hash used for integrity.
- **NACK:** negative acknowledgment.
- **Thread-per-client:** one server thread per connection.
- **RTT:** round-trip time.
- **Throughput:** bytes transferred per unit time.
- **TTA:** test-time augmentation.
- **Ensemble:** combine predictions from multiple models.
- **Patient hash:** pseudonymous patient/case identifier in the metadata.

---

## 15. What each teammate should contribute

### 15.1 Haneeth (model, CN, deployment, paper)

- Model section: EfficientNet-B0, transfer learning, transforms, loss, sampler, optimizer, AMP.
- Experiment tables: baseline, 2×2 factorial, focal/effective/schedule ablations.
- Figures: training curves, per-class F1 bars, confusion matrices (pristine/degraded).
- CN section: protocol, server/client, failure demos, latency/throughput plots.
- Deployment section: Streamlit screenshots and integration.
- Reproducibility: commands, configs, environment.
- Final `phase2_test` evaluation.

### 15.2 Nivrutha (quality/features, Cellpose, evaluation support, figures)

- Feature section: the 7 features, the 7 quality-feature CSVs (plus the processing manifest and Laplacian investigation), 63 plots, severity analysis, Laplacian audit.
- Verify the ported `src/quality/features.py` against her CSVs (already validated, but confirm).
- Degradation-type analysis: Laplacian vs median-filter residual, noise vs blur.
- Cellpose pilot: original/mask/overlay/crop, failure notes.
- Evaluation support: per-class tables, confusion matrices, plots from prediction CSVs.
- DIP/quality/Cellpose report sections and PPT slides.

### 15.3 Shared

- Agree on the final numbers and the exact wording of limitations.
- Cross-review the report for consistent counts, class order, and metric definitions.
- Rehearse the demo once end-to-end.

### 15.4 Suggested report structure

1. Introduction and objectives.
2. Dataset description: counts, splits, degradation, class imbalance.
3. EDA and quality features: distributions, plots, degradation-type insight.
4. Preprocessing and routing: noise vs blur, denoise/skip/deblur.
5. Model and methodology: EfficientNet-B0, transfer learning, imbalance handling.
6. Experiments and results: factorial table, ablations, per-class results, confusion matrices.
7. Analysis: mixed-domain, focal, rare-class bottleneck, resolution/batch, overfitting.
8. CN design and implementation: architecture, protocol, demos, measurements.
9. Deployment: Streamlit.
10. Limitations and future work.
11. Conclusion.
12. References.

### 15.5 Figures and tables to include

- Class distribution bar chart (log scale) with support counts.
- Split counts table.
- Feature distributions by severity/type.
- Scatter: Laplacian variance vs median-filter residual, colored by degradation type.
- Training curves (loss and macro-F1) for the best runs.
- 2×2 factorial results table.
- Per-class F1 bar chart for the best model, pristine vs degraded.
- Confusion matrices for the best model, pristine and degraded.
- CN frame format diagram and sequence diagram.
- CN latency/throughput plots.
- Streamlit screenshot.

### 15.6 Mapping to the existing documentation checklist

Use the files in `../documentation_checklist/` as the source for:

- `01_project_overview.md`: project summary.
- `03_dataset_description.md`: dataset counts and classes.
- `04_dataset_structure_and_metadata.md`: splits, folders, metadata.
- `06_local_dataset_verification.md`: local counts and resolutions.
- `07_eda_class_distribution.md`: class imbalance.
- `08_eda_visual_inspection.md`: visual examples.
- `09_eda_resolution_channels_quality.md`: resolution/channel/quality.
- `11_dip_workflow.md`: DIP pipeline.
- `12_cn_integration.md`: CN design.
- `13_current_status_and_next_work.md`: status and next steps.

Do not duplicate; link and extend.

---

## 16. Exact commands

### 16.1 Environment

```bash
cd "/run/media/_a_h/New Volume/clg/IIIT RAICHUR/Project_DIP/DIP/training"
source .venv/bin/activate
```

If installing packages, use:

```bash
uv pip install --python .venv/bin/python <package>
```

### 16.2 Build the manifest

```bash
python scripts/build_manifests.py
```

### 16.3 Validate the features

```bash
python scripts/validate_features.py --n 50
python -m pytest tests/test_features.py -q
```

### 16.4 Best model (focal) training

```bash
python scripts/train.py \
  --train-split phase1_train,phase2_train \
  --train-folder pristine,degraded \
  --val-split phase2_eval \
  --val-folder pristine,degraded \
  --img-size 224 --batch-size 64 --lr 5e-4 \
  --epochs 10 --sampler sqrt --loss focal \
  --run-name effb0_224_bs64_mixed_focal --num-workers 4
```

### 16.5 Evaluate on pristine and degraded

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/effb0_224_bs64_mixed_focal/best.pt \
  --split phase2_eval --folder pristine \
  --out-dir results/effb0_224_bs64_mixed_focal/eval_pristine

python scripts/evaluate.py \
  --checkpoint checkpoints/effb0_224_bs64_mixed_focal/best.pt \
  --split phase2_eval --folder degraded \
  --out-dir results/effb0_224_bs64_mixed_focal/eval_degraded
```

### 16.6 Final test evaluation (to do)

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/effb0_224_bs64_mixed_focal/best.pt \
  --split phase2_test --folder pristine \
  --out-dir results/effb0_224_bs64_mixed_focal/test_pristine

python scripts/evaluate.py \
  --checkpoint checkpoints/effb0_224_bs64_mixed_focal/best.pt \
  --split phase2_test --folder degraded \
  --out-dir results/effb0_224_bs64_mixed_focal/test_degraded
```

### 16.7 Download HF metadata (requires auth)

```bash
hf auth login
python scripts/download_metadata.py
python scripts/build_manifests.py
```

---

## 17. One-line takeaway

Mixed-domain training makes the model robust to degraded images, focal loss improves the hard mid-frequency classes, and the remaining weakness is concentrated in a few visually overlapping ultra-rare classes. The CN and deployment components are designed but not yet fully implemented, and the final held-out test evaluation plus the patient-level leakage check are the next must-dos.
