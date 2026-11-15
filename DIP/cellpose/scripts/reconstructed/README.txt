Cellpose Pilot Scripts — Reconstructed Helpers

Contents
1. cellpose_pilot_settings_audit.py
   Reports the Cellpose version and callable signatures in the environment
   where it is run. It cannot establish historical settings from an earlier
   Colab session.

2. cellpose_pilot_postprocessing_rebuild.py
   Rebuilds basic observations and an image/crop/overlay manifest from an
   existing Cellpose results folder. It does not run segmentation and does
   not reproduce every original analysis table or comparison grid.

Important limitations
- These are reconstructed helper scripts, NOT verified exact copies of the
  original notebook cells.
- The post-processing script computes a normalized centroid distance from
  the image center from available masks. Its formula may differ from the
  original calculation used to create distance_from_center_normalized.
- It does not regenerate cellpose_crop_quality_features.csv,
  selected_mask_verification.csv, or comparison grids.
- Set PILOT_ROOT in the post-processing script to the folder containing
  originals/, masks/, crops/, and overlays/.
- Run in an environment with numpy, pandas, Pillow, and opencv-python installed.
