# Pet Breed Classifier — 小樣本寵物品種辨識（MobileNetV2 + Streamlit）

本專案示範一個完整的深度學習影像分類流程：使用 **Transfer Learning（MobileNetV2 / ImageNet 預訓練）** 在 **小樣本資料（每類 5 張）** 的情境下完成寵物品種辨識，並延伸為可部署的 **Streamlit Web App**。

你可以上傳寵物照片，系統會輸出：

* Top-K 預測結果與信心分數
* Grad-CAM 可解釋性視覺化（顯示模型關注區域）
* 訓練端可產生 Learning Curves、Confusion Matrix、Classification Report

---

## 專案特色

* **小樣本可 demo**：每個品種只需少量圖片（例如每類 5 張）即可完成展示版模型
* **Transfer Learning**：採用 MobileNetV2 預訓練權重，避免從零訓練造成嚴重 overfitting
* **部署友善**：模型檔案小於 GitHub 25MB 限制，適合直接部署到 Streamlit Cloud
* **可解釋性**：提供 Grad-CAM（建議使用 `.pth state_dict` 版本以穩定支援 hook）

---

## 資料夾結構

```
Pet-Breed-Classifier/
├── dataset/                     # 原始資料：每類一個資料夾（每類約 5 張）
├── data_split/                  # 自動切分後資料：train/val/test
├── notebooks/
│   └── train_pet_breed.ipynb    # Colab/Notebook：切分、訓練、評估、匯出模型
├── app/
│   ├── streamlit_app.py         # Streamlit Web App
│   ├── model/
│   │   ├── model_mobilenetv2.pth        
│   │   ├── model_mobilenetv2.pt         
│   │   └── class_names.json
│   └── utils/
│       ├── modeling.py
│       ├── preprocess.py
│       └── gradcam.py
├── REPORT_ABSTRACT.txt          # 300 字英文摘要
├── agent_dev_log.md             # Agent 開發過程對話紀錄
├── requirements.txt
└── README.md
```

---

## 1) 準備資料（dataset）

請將圖片放入 `dataset/`，以資料夾名稱作為類別名稱，例如：

```
dataset/
  American Shorthair/
    img1.jpg
    img2.jpg
    ...
  Shiba Inu/
    shiba1.jpg
    ...
```

建議每類至少 5 張（本專案預設切分 3/1/1）。

---

## 2) 訓練與匯出模型（Colab / Notebook）

開啟並執行：

* `notebooks/train_pet_breed.ipynb`

Notebook 會完成：

1. 將 `dataset/` 自動切分為 `data_split/train|val|test`（每類 3/1/1）
2. 使用 MobileNetV2 進行 Transfer Learning 訓練
3. 產生：

   * Learning Curves（Loss/Accuracy）
   * Confusion Matrix
   * Classification Report
4. 匯出模型與類別檔案到 `exports/`

### 匯出檔案（Notebook 產生）

```
exports/
  model_mobilenetv2.pth
  class_names.json
  (可選) model_mobilenetv2.pt
```

## 3) 本機執行 Streamlit

在專案根目錄安裝套件並啟動：

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 4) 部署到 Streamlit Cloud（streamlit.app）

1. 將整個 repo push 到 GitHub（請確認模型檔 `< 25MB`）
2. 到 Streamlit Cloud 新增 App：

   * **Main file path**：`app/streamlit_app.py`

### 部署前必檢查

請確保 GitHub repo 內存在：

* `app/model/model_mobilenetv2.pth`
* `app/model/class_names.json`
* `requirements.txt`

---

## 6) Grad-CAM 說明

* **TorchScript (`.pt`) 版本** 在部分雲端環境可能無法使用 hook，因此 Grad-CAM 可能產生失敗。
* 建議使用 **state_dict (`.pth`) 版本**：Streamlit 端重新 build MobileNetV2 再載入權重，Grad-CAM 最穩定。

---

## 7) 作業繳交對應

* **300 字摘要**：`REPORT_ABSTRACT.txt`
* **Agent 對話紀錄**：`agent_dev_log.md`
* **GitHub + Streamlit.app**：本 repo + Streamlit Cloud 連結

