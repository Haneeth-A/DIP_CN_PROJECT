
import torch
from cellpose import models

print("GPU:", torch.cuda.get_device_name(0))

model = models.CellposeModel(
    gpu=True,
    pretrained_model="cpsam"
)

print("Cellpose model loaded successfully.")


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm

base_dir = "/content/cellpose_pilot"

folders = ["masks", "overlays", "masked_images", "crops"]

for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

pilot = pd.read_csv(
    "/content/cellpose_balanced_pilot_sample_list.csv",
    dtype={"image_id": str}
)

existing_dir = "/content/existing_cellpose_originals"
existing_ids = set()

for root, dirs, files in os.walk(existing_dir):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            existing_ids.add(
                f.split("_")[-1].split(".")[0].zfill(8)
            )

pilot["image_id_fixed"] = pilot["image_id"].str.zfill(8)

to_process = pilot[
    ~pilot["image_id_fixed"].isin(existing_ids)
].copy()

observations = []
failures = []

for index, row in to_process.iterrows():

    image_id = row["image_id_fixed"]

    try:
        image_matches = glob.glob(
            f"{base_dir}/originals/{image_id}.*"
        )

        if not image_matches:
            raise FileNotFoundError("Original image not found")

        image_path = image_matches[0]

        img = np.array(
            Image.open(image_path).convert("RGB")
        )

        masks, flows, styles = model.eval(
            img,
            diameter=None,
            cellprob_threshold=0.0
        )

        labels, counts = np.unique(
            masks[masks > 0],
            return_counts=True
        )

        object_count = len(labels)

        # Save complete mask
        np.save(
            f"{base_dir}/masks/{image_id}.npy",
            masks
        )

        # Create coloured mask
        colored_mask = np.zeros(
            (*masks.shape, 3),
            dtype=np.uint8
        )

        for label in labels:
            colored_mask[masks == label] = (
                np.array(cm.nipy_spectral(
                    label / max(object_count, 1)
                )[:3]) * 255
            ).astype(np.uint8)

        Image.fromarray(colored_mask).save(
            f"{base_dir}/masks/{image_id}.png"
        )

        # Save overlay
        overlay = (
            img.astype(float) * 0.55 +
            colored_mask.astype(float) * 0.45
        ).clip(0, 255).astype(np.uint8)

        Image.fromarray(overlay).save(
            f"{base_dir}/overlays/{image_id}.png"
        )

        # Select largest detected object as candidate WBC
        if object_count > 0:

            largest_label = labels[np.argmax(counts)]
            selected_mask = masks == largest_label

            area = int(selected_mask.sum())
            area_percent = area / selected_mask.size * 100

            masked_img = img.copy()
            masked_img[~selected_mask] = 0

            Image.fromarray(masked_img).save(
                f"{base_dir}/masked_images/{image_id}.png"
            )

            ys, xs = np.where(selected_mask)

            padding = 20

            x1 = max(0, xs.min() - padding)
            x2 = min(img.shape[1], xs.max() + padding + 1)
            y1 = max(0, ys.min() - padding)
            y2 = min(img.shape[0], ys.max() + padding + 1)

            crop = masked_img[y1:y2, x1:x2]

            Image.fromarray(crop).save(
                f"{base_dir}/crops/{image_id}.png"
            )

        else:
            area = 0
            area_percent = 0

            Image.fromarray(img).save(
                f"{base_dir}/masked_images/{image_id}.png"
            )

            Image.fromarray(img).save(
                f"{base_dir}/crops/{image_id}.png"
            )

        observations.append({
            "image_id": int(image_id),
            "class": row["class_label"],
            "severity": row["severity_label"],
            "split": row["split"],
            "width": img.shape[1],
            "height": img.shape[0],
            "detected_objects": object_count,
            "selected_mask_area_pixels": area,
            "mask_area_percent": area_percent,
            "visual_assessment": "Needs manual inspection",
            "observations": ""
        })

    except Exception as e:
        failures.append({
            "image_id": image_id,
            "error": str(e)
        })

    if (len(observations) + len(failures)) % 10 == 0:
        print(
            "Processed:",
            len(observations) + len(failures),
            "/ 80"
        )

# Save results
observations_df = pd.DataFrame(observations)

observations_df.to_csv(
    f"{base_dir}/observations_new_80.csv",
    index=False
)

pd.DataFrame(failures).to_csv(
    f"{base_dir}/failed_images.csv",
    index=False
)

print("\nPROCESSING COMPLETE")
print("Expected:", len(to_process))
print("Successful:", len(observations))
print("Failed:", len(failures))
print("Crops saved:", len(os.listdir(f"{base_dir}/crops")))