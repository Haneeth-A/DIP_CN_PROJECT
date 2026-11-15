# WBC Cancer Prediction Project — Source Code

## Project overview

This source package contains code for image-quality feature extraction,
severity-wise descriptive analysis, and pilot WBC image segmentation.

Dataset: `Xin-Tian/wbcbench2026`

The project covers 13 WBC classes.

## 1. Quality feature extraction

The quality extraction notebook calculates seven image-level features:

1. `laplacian_variance` — variance of the grayscale Laplacian, used as a focus/sharpness indicator.
2. `median_filter_residual` — mean absolute difference between the grayscale image and its 3x3 median-filtered version.
3. `brightness` — mean grayscale intensity.
4. `contrast` — standard deviation of grayscale intensity.
5. `extreme_pixel_percentage` — percentage of grayscale pixels with values <= 5 or >= 250.
6. `rgb_colour_variation` — mean of the per-channel spatial standard deviations in RGB.
7. `edge_density` — percentage of edge pixels detected using Canny thresholds 100 and 200.

These are image-quality descriptors. They are not diagnostic measurements
and do not independently predict cancer.

### Seven separate output CSVs

| Dataset | Expected rows |
|---|---:|
| Phase 1 train pristine | 8,288 |
| Phase 2 train pristine | 24,897 |
| Phase 2 eval pristine | 5,350 |
| Phase 2 test pristine | 16,467 |
| Phase 2 train degraded | 24,897 |
| Phase 2 eval degraded | 5,350 |
| Phase 2 test degraded | 16,467 |

The CSV files retain separate dataset splits and source types.

### Expected CSV columns

`image_id`, `split`, `class_label`, `severity_label`, followed by the
seven quality features listed above.

## 2. Severity analysis

The notebook creates:
- Severity distribution summary.
- Quality feature mean and standard deviation by severity.
- Seven severity distribution plots.
- Twenty-one degraded-data feature comparison plots.
- Thirty-five additional comparison plots.

Expected total: 63 PNG plots and two principal summary CSV files.

The additional comparisons include pristine-versus-degraded Phase 2
comparisons, Phase 1-versus-Phase 2 pristine train comparisons, and
degraded train/evaluation/test comparisons.

## 3. Dataset input structure

The quality notebook expects the extracted dataset under:

`/content/wbcbench2026`

Expected subfolders:
- `pristine/`
- `degraded/`

The Parquet filenames and shard structure are defined in the extraction
notebook. Make sure all required Parquet files are present before running
the extraction cell.

The source code does not contain a Hugging Face access token. If the
dataset requires authentication, authenticate privately in Colab.
Never place a personal access token directly in a shared notebook.

## 4. Cellpose pilot segmentation

The Cellpose source notebook documents the pilot segmentation workflow.

Recorded configuration:
- Cellpose version: 4.2.1.1
- Model class: `CellposeModel`
- Pretrained model: `cpsam`
- GPU setting in the original run: `gpu=True`
- Evaluation: `diameter=None`, `cellprob_threshold=0.0`
- Original execution environment: Google Colab with a T4 GPU.

The pretrained CPSAM weights are obtained through Cellpose model
initialization. No custom Cellpose model was trained and no custom
trained weights are included in this package.

The continuation processing cell expects:
- `/content/cellpose_balanced_pilot_sample_list.csv`
- `/content/cellpose_pilot/originals/`
- `/content/existing_cellpose_originals/`

The original pilot sample-selection procedure was not present in the
audited notebook cells. A reconstructed pilot list was available, but
its split and source-CSV metadata were not populated. Do not interpret
the list as proof of a particular random-sampling procedure.

The completed pilot output contains 100 images. The segmentation code
is included for source-code documentation and reproducibility; the
completed segmentation does not need to be rerun to use the existing
result files.

## 5. Important limitations

- Segmentation mask integrity checks do not establish biological
  segmentation accuracy.
- The 100-image pilot was not validated against expert-annotated
  ground-truth masks.
- Visual inspection remains necessary.
- Quality features are descriptive image measurements, not a cancer
  diagnosis or a trained cancer-classification model.
- No severity prediction model or custom model weights were trained
  by the quality-feature analysis.

## 6. Software requirements

Main Python packages:
- numpy
- pandas
- opencv-python-headless
- pyarrow
- matplotlib
- tqdm
- Pillow
- cellpose
- torch

For the Cellpose notebook, install Cellpose in the notebook environment.
GPU availability depends on the runtime.

## 7. Running the notebooks

Recommended environment: Google Colab.

1. Upload or mount the dataset in the expected directory.
2. Open the quality notebook and run its cells in order.
3. Check the CSV validation output before continuing to severity analysis.
4. Open the Cellpose notebook only when its required input files are available.
5. Keep the original result ZIP files separate from this source-code package.

## Status

**DONE.** The notebook produced the seven feature CSVs and the severity
analysis. The seven-feature implementation was ported to
`DIP/training/src/quality/features.py` and validated (9 passing tests).

## Pending / balance

- The Cellpose pilot was not validated against expert masks.
- No severity/triage model was trained.
- The notebook expects `/content/wbcbench2026` (Colab paths); adjust for local
  runs.
- Authentication for the dataset (if required) must never be committed.
