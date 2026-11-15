
import sys
import io
import tempfile
import threading
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "DIP" / "training" / "src"))
sys.path.insert(0, str(REPO_ROOT / "CN" / "src"))

EDGE_MODEL = (
    REPO_ROOT / "CN/models/edge_deployment/model_int8.onnx"
)
EDGE_LABELS = (
    REPO_ROOT / "CN/models/edge_deployment/labels.json"
)
EDGE_PREPROCESSING = (
    REPO_ROOT / "CN/models/edge_deployment/preprocessing.json"
)
POLICY_PATH = (
    REPO_ROOT / "CN/results/cn_edge_fog/policy_tree.joblib"
)

FOG_CHECKPOINTS = [
    REPO_ROOT / "DIP/training/checkpoints/"
    "effb0_224_bs64_mixed_focal_reran/best.pt",

    REPO_ROOT / "DIP/training/checkpoints/"
    "effb0_224_bs64_mixed_eff/best.pt",

    REPO_ROOT / "DIP/training/checkpoints/"
    "effb0_224_bs64_mixed/best.pt",
]

# --------------------------------------------------
# STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="WBC Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 WBC Cancer Detection System")

st.write(
    "Upload a White Blood Cell image to perform image-quality "
    "assessment and Edge/Fog-based classification."
)

st.info(
    "Research prototype: predictions are experimental and "
    "must not be used as a clinical diagnosis."
)

# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------

try:
    from cn.edge.local_predictor import OnnxEdgePredictor
    from cn.edge.policy import LearnedPolicy
    from cn.edge.profiles import get
    from cn.client import FogClient
    from cn.server import FogServer
    from cn.fog_backend import TorchFogBackend

    from quality.features import extract_image_features

except Exception as e:
    st.error(f"Unable to import project modules: {e}")
    st.stop()


# --------------------------------------------------
# LOAD EDGE, POLICY AND FOG SERVER ONCE
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading Edge and Fog models...")
def load_pipeline():

    required_paths = [
        EDGE_MODEL,
        EDGE_LABELS,
        EDGE_PREPROCESSING,
        POLICY_PATH,
        *FOG_CHECKPOINTS
    ]

    missing = [str(p) for p in required_paths if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Required model files are missing:\n" + "\n".join(missing)
        )

    # Load INT8 Edge model
    edge = OnnxEdgePredictor(
        model_path=str(EDGE_MODEL),
        labels_path=str(EDGE_LABELS),
        preprocessing_path=str(EDGE_PREPROCESSING),
        predict_script=str(
            REPO_ROOT / "CN/models/edge_deployment/predict.py"
        ),
        threads=1
    )

    # Load saved learned routing policy
    policy = LearnedPolicy.load(str(POLICY_PATH))

    # Load three-checkpoint Fog ensemble
    fog_backend = TorchFogBackend(
        [str(p) for p in FOG_CHECKPOINTS],
        device="cpu",
        tta="full",
        model_name="fog_ensemble",
        num_threads=2,
        warmup=False
    )

    # Start real TCP Fog server
    server = FogServer(
        "127.0.0.1",
        0,
        backend=fog_backend,
        timeout_s=180.0
    )

    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True
    )

    server_thread.start()

    host, port = server.server_address

    # Create TCP client
    fog_client = FogClient(
        host=host,
        port=port,
        timeout_s=180.0,
        connect_timeout_s=10.0
    )

    return {
        "edge": edge,
        "policy": policy,
        "fog_client": fog_client,
        "server": server,
        "server_thread": server_thread,
        "fog_backend": fog_backend
    }


# --------------------------------------------------
# HELPER: TOP-3 PREDICTIONS
# --------------------------------------------------

def format_top3(items):

    rows = []

    for item in items:

        if isinstance(item, dict):
            label = item.get(
                "label",
                item.get("class", item.get("predicted_class", "Unknown"))
            )

            score = item.get(
                "confidence",
                item.get("probability", item.get("score", 0))
            )

        else:
            label, score = item[0], item[1]

        rows.append({
            "WBC Class": label,
            "Confidence": round(float(score), 6)
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# NETWORK PROFILE
# --------------------------------------------------

st.divider()
st.subheader("1. Network Configuration")

profile_options = {
    "WiFi": "wifi",
    "5G": "5g",
    "Poor WiFi": "poor_wifi",
    "LPWAN": "lpwan"
}

selected_profile = st.selectbox(
    "Select network profile",
    list(profile_options.keys()),
    index=0,
    help="The saved learned policy uses this network profile "
         "to decide whether to use Edge or Fog."
)

profile_name = profile_options[selected_profile]

st.caption(f"Selected profile: {profile_name}")


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.divider()
st.subheader("2. Upload WBC Image")

uploaded_file = st.file_uploader(
    "Choose a WBC image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG or PNG image."
)

if uploaded_file is None:

    st.info("Please upload a WBC image to continue.")

else:

    try:
        image_bytes = uploaded_file.getvalue()

        # Verify image
        image_check = Image.open(io.BytesIO(image_bytes))
        image_format = image_check.format
        image_check.verify()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        if image.width < 10 or image.height < 10:
            st.error("Image dimensions are too small.")

        else:

            st.success("Image uploaded successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image,
                    caption="Uploaded WBC Image",
                    width="stretch"
                )

            with col2:
                st.write("### Image Information")
                st.write(f"**File name:** {uploaded_file.name}")
                st.write(f"**Image format:** {image_format}")
                st.write(f"**Width:** {image.width} pixels")
                st.write(f"**Height:** {image.height} pixels")
                st.write("**Color channels:** 3 (RGB)")
                st.write(
                    f"**File size:** {len(image_bytes) / 1024:.2f} KB"
                )

            st.divider()

            # --------------------------------------------------
            # ANALYSIS BUTTON
            # --------------------------------------------------

            if st.button(
                "Analyze Image",
                type="primary",
                use_container_width=True
            ):

                temp_path = None

                try:
                    # Load all models and start Fog server
                    pipeline = load_pipeline()

                    edge = pipeline["edge"]
                    policy = pipeline["policy"]
                    fog_client = pipeline["fog_client"]

                    # Save uploaded image temporarily
                    suffix = Path(uploaded_file.name).suffix.lower()

                    if suffix not in [".jpg", ".jpeg", ".png"]:
                        suffix = ".jpg"

                    with tempfile.NamedTemporaryFile(
                        suffix=suffix,
                        delete=False
                    ) as temp_file:

                        temp_file.write(image_bytes)
                        temp_path = Path(temp_file.name)

                    st.divider()
                    st.subheader("3. Image Quality Assessment")

                    # Extract seven quality features
                    features = extract_image_features(
                        str(temp_path)
                    )

                    quality_df = pd.DataFrame(
                        [
                            {
                                "Quality Feature": key,
                                "Value": round(float(value), 6)
                            }
                            for key, value in features.items()
                        ]
                    )

                    st.dataframe(
                        quality_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # --------------------------------------------------
                    # EDGE INFERENCE
                    # --------------------------------------------------

                    st.divider()
                    st.subheader("4. Edge AI Prediction")

                    with st.spinner("Running local INT8 Edge model..."):
                        edge_result = edge.predict(str(temp_path))

                    edge_class = edge_result["predicted_class"]
                    edge_confidence = float(edge_result["confidence"])

                    edge_col1, edge_col2, edge_col3 = st.columns(3)

                    edge_col1.metric(
                        "Edge predicted class",
                        edge_class
                    )

                    edge_col2.metric(
                        "Edge confidence",
                        f"{edge_confidence:.4f}"
                    )

                    edge_col3.metric(
                        "Edge inference latency",
                        f"{edge_result['latency_ms']:.2f} ms"
                    )

                    st.write("**Edge Top-3 predictions**")

                    st.dataframe(
                        format_top3(edge_result["top3"]),
                        use_container_width=True,
                        hide_index=True
                    )

                    # --------------------------------------------------
                    # LEARNED POLICY
                    # --------------------------------------------------

                    st.divider()
                    st.subheader("5. Learned Edge/Fog Routing")

                    network_profile = get(profile_name)

                    decision = policy.decide(
                        features,
                        edge_result,
                        network_profile
                    )

                    action = decision.value.lower()

                    if action == "local":

                        st.success(
                            "LOCAL: The learned policy selected "
                            "the Edge prediction."
                        )

                        final_result = edge_result
                        prediction_source = "Edge INT8"
                        total_latency = float(
                            edge_result["latency_ms"]
                        )

                        fog_result = None

                    elif action == "offload":

                        st.warning(
                            "OFFLOAD: The learned policy selected "
                            "the Fog server."
                        )

                        with st.spinner(
                            "Sending image to Fog through TCP..."
                        ):

                            fog_result = fog_client.classify(
                                str(temp_path),
                                image_id=uploaded_file.name
                            )

                        if not fog_result.get("integrity_ok", False):
                            raise RuntimeError(
                                "Fog image integrity verification failed."
                            )

                        final_result = fog_result
                        prediction_source = "Fog ensemble"
                        total_latency = float(
                            fog_result["total_ms"]
                        )

                    else:
                        raise RuntimeError(
                            f"Unknown routing decision: {action}"
                        )

                    # --------------------------------------------------
                    # FINAL RESULT
                    # --------------------------------------------------

                    st.divider()
                    st.subheader("6. Final WBC Classification Result")

                    final_class = final_result["predicted_class"]
                    final_confidence = float(
                        final_result["confidence"]
                    )

                    result_col1, result_col2, result_col3 = st.columns(3)

                    result_col1.metric(
                        "Final predicted class",
                        final_class
                    )

                    result_col2.metric(
                        "Final confidence",
                        f"{final_confidence:.4f}"
                    )

                    result_col3.metric(
                        "Prediction source",
                        prediction_source
                    )

                    st.write("**Final Top-3 predictions**")

                    st.dataframe(
                        format_top3(final_result["top3"]),
                        use_container_width=True,
                        hide_index=True
                    )

                    # --------------------------------------------------
                    # PERFORMANCE DETAILS
                    # --------------------------------------------------

                    st.divider()
                    st.subheader("7. Performance and Communication")

                    perf_col1, perf_col2 = st.columns(2)

                    perf_col1.metric(
                        "Selected network",
                        selected_profile
                    )

                    perf_col2.metric(
                        "Total prediction latency",
                        f"{total_latency:.2f} ms"
                    )

                    if fog_result is not None:

                        st.write("### TCP Communication Details")

                        tcp_col1, tcp_col2, tcp_col3 = st.columns(3)

                        tcp_col1.metric(
                            "Upload",
                            f"{fog_result['upload_ms']:.2f} ms"
                        )

                        tcp_col2.metric(
                            "Fog inference",
                            f"{fog_result['latency_ms']:.2f} ms"
                        )

                        tcp_col3.metric(
                            "Download",
                            f"{fog_result['download_ms']:.2f} ms"
                        )

                        st.write(
                            f"**Bytes sent:** "
                            f"{fog_result['bytes_sent']}"
                        )

                        st.write(
                            f"**Bytes received:** "
                            f"{fog_result['bytes_received']}"
                        )

                        st.write(
                            "**TCP image integrity:** Verified"
                        )

                    else:

                        st.info(
                            "The image was classified locally. "
                            "No Fog communication was required."
                        )

                    st.divider()

                    st.caption(
                        "Experimental WBC classification system. "
                        "This result is not a medical diagnosis."
                    )

                except Exception as e:

                    st.error("Image analysis failed.")
                    st.exception(e)

                finally:

                    if temp_path is not None and temp_path.exists():
                        temp_path.unlink()

    except Exception as e:

        st.error(
            "Unable to read this image. "
            "Please upload a valid JPG, JPEG or PNG file."
        )

        st.exception(e)
