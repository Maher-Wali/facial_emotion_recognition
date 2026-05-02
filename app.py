"""
Facial Emotion Recognition — Streamlit Demo
Run: streamlit run app.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_EMOJI = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😄',
    'Neutral':  '😐',
    'Sad':      '😢',
    'Surprise': '😲',
}

IMG_SIZE = 112
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Model definitions (mirrors train.py) ──────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


def get_efficientnet(num_classes=7):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str, phase: int):
    if model_name == "baseline_cnn":
        model = CustomCNN(num_classes=7)
    else:
        model = get_efficientnet(num_classes=7)

    ckpt_path = os.path.join(RESULTS_DIR, model_name, f"best_phase{phase}.pth")
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("RGB")
    return val_tf(img).unsqueeze(0).to(DEVICE)


def apply_clahe_preview(pil_img: Image.Image) -> Image.Image:
    """Return a side-by-side CLAHE preview (grayscale)."""
    gray = np.array(pil_img.convert("L"))
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)


@torch.no_grad()
def predict(model, tensor: torch.Tensor):
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred   = int(probs.argmax())
    return pred, probs


def read_test_results(model_name: str) -> dict:
    path = os.path.join(RESULTS_DIR, model_name, "test_results.txt")
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    results[k.strip()] = v.strip()
    return results


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😄",
    layout="wide",
)

st.title("Facial Emotion Recognition")
st.caption("Custom CNN vs EfficientNet-B0 — trained on FERplus + RAF-DB")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox(
        "Architecture",
        ["efficientnet", "baseline_cnn"],
        format_func=lambda x: "EfficientNet-B0" if x == "efficientnet" else "Baseline CNN",
    )
    phase = st.radio(
        "Checkpoint",
        [1, 2],
        format_func=lambda p: f"Phase {p} — {'FERplus pre-train' if p == 1 else 'RAF-DB fine-tune'}",
        index=1,
    )

    st.divider()
    res = read_test_results(model_name)
    if res:
        st.metric("Test Accuracy", res.get("Test Accuracy", "—"))
        st.metric("Macro F1", res.get("Macro F1-score", "—"))
    st.caption(f"Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    model = load_model(model_name, phase)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_demo, tab_results = st.tabs(["Demo", "Model Results"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Demo
# ══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    input_mode = st.radio("Input source", ["Upload image", "Webcam"], horizontal=True)

    pil_img = None

    if input_mode == "Upload image":
        uploaded = st.file_uploader(
            "Choose a face image", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        if uploaded:
            pil_img = Image.open(uploaded)

    else:
        cam_img = st.camera_input("Take a photo")
        if cam_img:
            pil_img = Image.open(cam_img)

    if pil_img is not None:
        col_img, col_pred = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("Input")
            st.image(pil_img, width="stretch")
            with st.expander("CLAHE preview (preprocessing)"):
                clahe_img = apply_clahe_preview(pil_img)
                st.image(clahe_img, caption="Grayscale + CLAHE (112×112)", width="stretch")

        with col_pred:
            st.subheader("Prediction")
            tensor      = preprocess(pil_img)
            pred_idx, probs = predict(model, tensor)
            emotion     = EMOTIONS[pred_idx]
            confidence  = float(probs[pred_idx])
            emoji       = EMOTION_EMOJI[emotion]

            st.markdown(
                f"<div style='text-align:center; font-size:4rem'>{emoji}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h2 style='text-align:center'>{emotion}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='text-align:center; color:gray'>Confidence: {confidence:.1%}</p>",
                unsafe_allow_html=True,
            )

            st.divider()
            st.caption("All class probabilities")
            prob_df = pd.DataFrame({
                "Emotion": EMOTIONS,
                "Probability": probs.tolist(),
            }).sort_values("Probability", ascending=True)
            st.bar_chart(prob_df.set_index("Emotion"), horizontal=True, color="#4C72B0")

    else:
        st.info("Upload an image or take a webcam photo to get a prediction.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Results
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    model_dir = os.path.join(RESULTS_DIR, model_name)
    label     = "EfficientNet-B0" if model_name == "efficientnet" else "Baseline CNN"
    st.subheader(f"Results — {label}")

    # ── Training curves ────────────────────────────────────────────────────────
    col_loss, col_acc = st.columns(2)
    loss_img = os.path.join(model_dir, "loss_curves.png")
    acc_img  = os.path.join(model_dir, "accuracy_curves.png")
    if os.path.exists(loss_img):
        col_loss.image(loss_img, caption="Loss curves", width="stretch")
    if os.path.exists(acc_img):
        col_acc.image(acc_img, caption="Accuracy curves", width="stretch")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    st.divider()
    cm_img = os.path.join(model_dir, "confusion_matrix.png")
    if os.path.exists(cm_img):
        cm_col, _ = st.columns([1, 1])
        cm_col.image(cm_img, caption="Confusion matrix (RAF-DB test set)", width="stretch")

    # ── Classification report ──────────────────────────────────────────────────
    st.divider()
    st.caption("Per-class metrics (RAF-DB test set)")
    cr_path = os.path.join(model_dir, "classification_report.csv")
    if os.path.exists(cr_path):
        cr_df = pd.read_csv(cr_path)
        # highlight macro/weighted avg rows
        def highlight_avg(row):
            if row["class"] in ("macro avg", "weighted avg"):
                return ["background-color: #f0f0f0"] * len(row)
            return [""] * len(row)
        st.dataframe(
            cr_df.style.apply(highlight_avg, axis=1).format(
                {"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}
            ),
            hide_index=True,
        )

    # ── Epoch logs ─────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Training logs (per epoch)"):
        p1_log = os.path.join(model_dir, "phase1_log.csv")
        p2_log = os.path.join(model_dir, "phase2_log.csv")
        if os.path.exists(p1_log):
            st.caption("Phase 1 — FERplus")
            st.dataframe(pd.read_csv(p1_log), hide_index=True)
        if os.path.exists(p2_log):
            st.caption("Phase 2 — RAF-DB")
            st.dataframe(pd.read_csv(p2_log), hide_index=True)
