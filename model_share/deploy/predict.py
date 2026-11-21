#!/usr/bin/env python
"""INT8 edge inference wrapper: ONNX Runtime CPU, batch size 1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "model_int8.onnx"
DEFAULT_LABELS = HERE / "labels.json"
DEFAULT_PREPROC = HERE / "preprocessing.json"


class Predictor:
    def __init__(
        self,
        model_path=None,
        labels_path=None,
        preprocessing_path=None,
        threads=1,
    ):
        import onnxruntime as ort

        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL
        self.labels_obj = json.loads(Path(labels_path or DEFAULT_LABELS).read_text())
        self.pre = json.loads(Path(preprocessing_path or DEFAULT_PREPROC).read_text())
        self.labels = self.labels_obj["labels"]
        self.img_size = int(self.pre["image_size"])
        self.mean = np.asarray(self.pre["normalize"]["mean"], dtype=np.float32)
        self.std = np.asarray(self.pre["normalize"]["std"], dtype=np.float32)

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = int(threads)
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Avoid U8S8 accumulator saturation on x64 CPUs without VNNI.
        options.add_session_config_entry("session.x64quantprecision", "1")
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.img_size, self.img_size),
            getattr(Image, "Resampling", Image).BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        array = (array - self.mean[:, None, None]) / self.std[:, None, None]
        return np.ascontiguousarray(array[None, ...], dtype=np.float32)

    def predict_array(self, array):
        logits = self.session.run([self.output_name], {self.input_name: array})[0]
        logits = np.asarray(logits, dtype=np.float32).reshape(1, -1)
        if logits.shape[1] != len(self.labels):
            raise ValueError(
                f"Expected {len(self.labels)} outputs, got {logits.shape[1]}"
            )
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs[0]

    def predict_image(self, image_path):
        array = self.preprocess(image_path)
        probs = self.predict_array(array)
        order = np.argsort(-probs)[:3]
        return {
            "image": str(image_path),
            "predicted_index": int(order[0]),
            "predicted_label": self.labels[int(order[0])],
            "confidence": float(probs[int(order[0])]),
            "top3": [
                {
                    "index": int(index),
                    "label": self.labels[int(index)],
                    "confidence": float(probs[int(index)]),
                }
                for index in order
            ],
            "probabilities": [float(value) for value in probs],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--preprocessing", default=str(DEFAULT_PREPROC))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    predictor = Predictor(args.model, args.labels, args.preprocessing, args.threads)
    results = [predictor.predict_image(path) for path in args.image]
    if args.compact:
        for result in results:
            print(json.dumps({
                "image": result["image"],
                "predicted_label": result["predicted_label"],
                "confidence": result["confidence"],
                "top3": result["top3"],
            }, indent=2))
    else:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))


if __name__ == "__main__":
    main()
