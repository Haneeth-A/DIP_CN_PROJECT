
WBC CELLPOSE PILOT - SOURCE CODE

CONTENTS
1. cellpose_balanced_pilot_sample_list.csv
   Contains the selected 100 image IDs and their class, severity and split labels.

2. cellpose_remaining_80_segmentation.py
   Original segmentation code for the remaining 80 images using pretrained CPSAM.

3. cellpose_original_postprocessing.py
   Original post-processing code for:
   - Image crop manifest
   - Combining observations
   - Comparison grids
   - Crop quality features
   - Selected mask verification

MODEL AND SETTINGS
- Cellpose version: 4.2.1.1
- Model: CPSAM pretrained model
- GPU: Used in the original workflow
- Diameter: None
- Cell probability threshold: 0.0
- Largest detected object selected as candidate WBC
- Crop padding: 20 pixels

IMPORTANT LIMITATIONS
- The original segmentation script for the first 20 images was not found
  in the saved notebooks or Python scripts searched.
- The remaining-80 script is original notebook code.
- The post-processing script contains original notebook code.
- The sample list alone does not contain the original images.
  The corresponding images must be extracted from the source dataset.
- The scripts use the original Colab paths and may need path changes
  when run in another environment.
- The visual_assessment field is a placeholder saying
  "Needs manual inspection"; it does not indicate completed manual review.

DEPENDENCIES
See requirements.txt.

The pretrained CPSAM model is downloaded by Cellpose when initialized.
No custom model training or custom weights were used.
