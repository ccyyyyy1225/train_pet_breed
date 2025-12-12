import os
import sys

# 讓 Streamlit Cloud 能找到 app/utils
sys.path.append(os.path.dirname(__file__))

import json
import numpy as np
import streamlit as st
from PIL import Image
import torch

from utils.modeling import load_torchscript_model
from utils.preprocess import preprocess_image
from utils.gradcam import GradCAM, overlay_cam_on_image

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Pet Breed Classifier", layout="wide")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_mobilenetv2.pt")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.json")

st.title("寵物品種辨識 Demo（Transfer Learning / ResNet18）")
st.caption("上傳一張寵物照片，系統會輸出 Top-K 品種預測與信心分數。")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("設定")
    device = "cpu"  # Streamlit Cloud 通常是 CPU
    topk = st.slider("Top-K", min_value=1, max_value=5, value=3, step=1)
    use_gradcam = st.checkbox("顯示 Grad-CAM（可選）", value=False)
    st.markdown("---")
    st.markdown("**提示**：如果 Grad-CAM 顯示失敗，關掉即可（TorchScript hook 在部分環境可能不支援）。")

# -------------------------
# Check files
# -------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_PATH):
    st.error(
        "找不到模型或類別檔案。\n\n"
        "請確認 repo 內存在：\n"
        "- `app/model/model_resnet18.pt`\n"
        "- `app/model/class_names.json`\n\n"
        "若你剛從 Colab 匯出，請把 `exports/` 內的兩個檔案複製過來。"
    )
    st.stop()

# -------------------------
# Load model / classes
# -------------------------
@st.cache_resource
def load_assets():
    model = load_torchscript_model(MODEL_PATH, device=device)
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_assets()

# -------------------------
# Helper: predict
# -------------------------
def predict(pil_img: Image.Image, topk: int = 3):
    x = preprocess_image(pil_img, device=device, img_size=224)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

    k = min(topk, len(probs))
    idx = probs.argsort()[::-1][:k]
    results = [(class_names[i], float(probs[i]), int(i)) for i in idx]
    return results, probs

# -------------------------
# UI: uploader
# -------------------------
uploaded = st.file_uploader("上傳寵物照片（JPG / PNG）", type=["jpg", "jpeg", "png"])

col_img, col_res = st.columns([1, 1])

if uploaded is None:
    st.info("請先上傳一張寵物照片。")
    st.stop()

# Read image
pil_img = Image.open(uploaded).convert("RGB")

with col_img:
    st.subheader("輸入圖片")
    st.image(pil_img, use_container_width=True)

# Predict
results, full_probs = predict(pil_img, topk=topk)

top1_name, top1_prob, top1_idx = results[0]

with col_res:
    st.subheader("預測結果")
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:14px;border:1px solid #e5e7eb;">
            <div style="font-size:14px;color:#6b7280;">Top-1 Prediction</div>
            <div style="font-size:28px;font-weight:800;margin-top:4px;">{top1_name}</div>
            <div style="font-size:16px;margin-top:6px;">信心分數：<b>{top1_prob*100:.2f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.markdown("### Top-K 詳細列表")

    # Progress bars
    for name, prob, _ in results:
        st.write(f"**{name}**  —  {prob*100:.2f}%")
        st.progress(min(max(prob, 0.0), 1.0))

    # Optional: bar chart (clean)
    st.write("")
    st.markdown("### 視覺化（Top-K）")
    chart_data = {"breed": [r[0] for r in results], "confidence": [r[1] for r in results]}
    st.bar_chart(chart_data, x="breed", y="confidence")

# -------------------------
# Optional Grad-CAM
# -------------------------
if use_gradcam:
    st.divider()
    st.subheader("Grad-CAM（可解釋性視覺化）")
    st.caption("顯示模型關注的影像區域（例如耳朵、臉部、毛色等）。若環境不支援 hook 會自動跳過。")

    try:
        cam_engine = GradCAM(model, target_layer_name="layer4")
        # 需要可求梯度的輸入
        x_gc = preprocess_image(pil_img, device=device, img_size=224)
        x_gc = x_gc.clone().requires_grad_(True)

        cam = cam_engine.generate(x_gc, class_idx=top1_idx)
        overlay = overlay_cam_on_image(pil_img, cam, alpha=0.40)

        if overlay is None:
            st.warning("Grad-CAM 無法產生（TorchScript hook 可能不支援）。請先關閉此功能，或改用非 TorchScript 模型格式。")
        else:
            st.image(overlay, caption=f"Grad-CAM Overlay（Top-1: {top1_name}）", use_container_width=True)

    except Exception as e:
        st.warning(f"Grad-CAM 產生失敗：{e}")
