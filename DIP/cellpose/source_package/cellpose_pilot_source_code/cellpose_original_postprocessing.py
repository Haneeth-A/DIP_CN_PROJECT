

# =================================================================
# ORIGINAL NOTEBOOK CELL 244
# =================================================================

import os
import pandas as pd

base = "/content/cellpose_pilot"

pilot = pd.read_csv(
    "/content/cellpose_balanced_pilot_sample_list.csv",
    dtype={"image_id": str}
)

records = []

for _, row in pilot.iterrows():
    image_id = row["image_id"].zfill(8)

    # Find crop using either naming convention
    crop_matches = [
        f for f in os.listdir(f"{base}/crops")
        if f.startswith(image_id) or f.endswith(f"{image_id}.png")
    ]

    records.append({
        "image_id": image_id,
        "class_label": row["class_label"],
        "severity_label": row["severity_label"],
        "split": row["split"],
        "crop_path": f"crops/{crop_matches[0]}" if crop_matches else "",
        "crop_available": len(crop_matches) > 0
    })

manifest = pd.DataFrame(records)

manifest.to_csv(
    f"{base}/image_crop_manifest.csv",
    index=False
)

print("Manifest rows:", len(manifest))
print("Crop files available:", manifest["crop_available"].sum())
print("Missing crops:", (~manifest["crop_available"]).sum())


# =================================================================
# ORIGINAL NOTEBOOK CELL 245
# =================================================================

import pandas as pd
import os

base = "/content/cellpose_pilot"

old = pd.read_csv(
    "/content/old_cellpose_results/observations.csv",
    dtype={"image_id": str}
)

new = pd.read_csv(
    f"{base}/observations_new_80.csv",
    dtype={"image_id": str}
)

old["image_id"] = old["image_id"].str.replace(r"\.0$", "", regex=True).str.zfill(8)
new["image_id"] = new["image_id"].str.replace(r"\.0$", "", regex=True).str.zfill(8)

combined = pd.concat([old, new], ignore_index=True)

combined.to_csv(
    f"{base}/observations_all_100.csv",
    index=False
)

print("Total observation rows:", len(combined))
print("Unique image IDs:", combined["image_id"].nunique())
print("Missing values:", combined.isnull().sum().sum())
print("\nSeverity counts:")
print(combined["severity"].value_counts())
print("\nClass counts:")
print(combined["class"].value_counts())


# =================================================================
# ORIGINAL NOTEBOOK CELL 248
# =================================================================

import os
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

base = "/content/cellpose_pilot"
out = f"{base}/comparison_grids"
os.makedirs(out, exist_ok=True)

manifest = pd.read_csv(f"{base}/image_crop_manifest.csv", dtype={"image_id": str})

def find_file(folder, image_id):
    for f in os.listdir(f"{base}/{folder}"):
        if f.startswith(image_id) or f.endswith(f"{image_id}.png") or f.endswith(f"{image_id}.jpg"):
            return f"{base}/{folder}/{f}"
    return None

for start in range(0, len(manifest), 10):
    batch = manifest.iloc[start:start+10]
    panels = []

    for _, row in batch.iterrows():
        image_id = str(row["image_id"]).zfill(8)

        original = find_file("originals", image_id)
        overlay = find_file("overlays", image_id)
        crop = find_file("crops", image_id)

        if not all([original, overlay, crop]):
            continue

        images = []
        for path in [original, overlay, crop]:
            img = Image.open(path).convert("RGB")
            img.thumbnail((180, 180))
            canvas = Image.new("RGB", (190, 210), "white")
            canvas.paste(img, ((190-img.width)//2, 5))
            images.append(canvas)

        panel = Image.new("RGB", (570, 240), "white")
        for i, img in enumerate(images):
            panel.paste(img, (i*190, 0))

        draw = ImageDraw.Draw(panel)
        draw.text((5, 215), f"{image_id} | {row['class_label']} | {row['severity_label']}", fill="black")
        panels.append(panel)

    grid = Image.new("RGB", (570, len(panels)*240), "white")
    for i, panel in enumerate(panels):
        grid.paste(panel, (0, i*240))

    grid.save(f"{out}/comparison_{start//10+1:02d}.jpg", quality=90)

print("Comparison grids created:", len(os.listdir(out)))
print("Files:", sorted(os.listdir(out)))


# =================================================================
# ORIGINAL NOTEBOOK CELL 260
# =================================================================

import cv2
import numpy as np
import pandas as pd
import os

base = "/content/cellpose_pilot"

manifest = pd.read_csv(
    f"{base}/image_crop_manifest.csv",
    dtype={"image_id": str}
)

results = []

for _, row in manifest.iterrows():

    image_id = str(row["image_id"]).zfill(8)
    crop_path = os.path.join(base, row["crop_path"])

    image = cv2.imread(crop_path, cv2.IMREAD_COLOR)

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_variance = cv2.Laplacian(
        gray, cv2.CV_64F
    ).var()

    median_residual = np.mean(np.abs(
        gray.astype(np.float32) -
        cv2.medianBlur(gray, 3).astype(np.float32)
    ))

    brightness = np.mean(gray)
    contrast = np.std(gray)

    extreme_percentage = (
        np.sum((gray <= 5) | (gray >= 250))
        / gray.size * 100
    )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb_variation = np.mean(
        np.std(rgb.astype(np.float32), axis=(0, 1))
    )

    edges = cv2.Canny(gray, 100, 200)

    edge_density = (
        np.count_nonzero(edges) / edges.size * 100
    )

    results.append({
        "image_id": image_id,
        "split": row["split"],
        "class_label": row["class_label"],
        "severity_label": row["severity_label"],
        "laplacian_variance": laplacian_variance,
        "median_filter_residual": median_residual,
        "brightness": brightness,
        "contrast": contrast,
        "extreme_pixel_percentage": extreme_percentage,
        "rgb_colour_variation": rgb_variation,
        "edge_density": edge_density
    })

crop_features = pd.DataFrame(results)

output = f"{base}/cellpose_crop_quality_features.csv"
crop_features.to_csv(output, index=False)

print("CROP FEATURE EXTRACTION COMPLETE")
print("Images processed:", len(crop_features))
print("Unique image IDs:", crop_features["image_id"].nunique())
print("Missing values:", crop_features.isnull().sum().sum())
print("Output:", output)


# =================================================================
# ORIGINAL NOTEBOOK CELL 264
# =================================================================

import os
import numpy as np
import pandas as pd

base = "/content/cellpose_pilot"

mask_folder = f"{base}/masks"
observations = pd.read_csv(f"{base}/observations_all_100.csv", dtype={"image_id": str})

results = []

for _, row in observations.iterrows():
    image_id = str(row["image_id"]).zfill(8)

    mask_path = f"{mask_folder}/{image_id}.npy"

    if not os.path.exists(mask_path):
        continue

    mask = np.load(mask_path)

    labels, counts = np.unique(mask[mask > 0], return_counts=True)

    if len(labels) == 0:
        continue

    largest_label = labels[np.argmax(counts)]
    largest_area = counts.max()

    results.append({
        "image_id": image_id,
        "class": row["class"],
        "severity": row["severity"],
        "detected_objects": row["detected_objects"],
        "largest_mask_area": largest_area,
        "largest_mask_label": largest_label,
        "total_mask_area": np.count_nonzero(mask)
    })

selected = pd.DataFrame(results)

print("Images checked:", len(selected))
print("Images with no detected objects:", 100 - len(selected))
print("\nLargest-mask area statistics:")
print(selected["largest_mask_area"].describe().round(2))

print("\nSample results:")
display(selected.head(10))

selected.to_csv(
    f"{base}/selected_mask_verification.csv",
    index=False
)
