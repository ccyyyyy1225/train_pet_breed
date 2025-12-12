# Agent 開發過程對話紀錄

---

## 1. 選題與可行性評估
**Student：** 我想做「寵物品種辨識」，但我每個類別只有 5 張圖，能做嗎？  
**AI：** 可以做「小樣本 demo」。策略是用預訓練模型（ResNet/MobileNet）當 feature extractor，配合強資料增強與簡單分類 head。並用 3/1/1 切 train/val/test。

---

## 2. 資料結構與切分策略
**Student：** 我目前是 `dataset/<class>/...`，沒有 train/test/val。  
**AI：** 我們寫一個腳本自動建立 `data_split/train|val|test/<class>/...`，每類 5 張就採 3/1/1 分配，並固定 random seed 以利重現。

---

## 3. 模型設計
**Student：** 我需要自己訓練 CNN 嗎？  
**AI：** 不建議從零訓練，資料太少會嚴重 overfit。使用 ImageNet 預訓練 ResNet18：  
- 先凍結 backbone，只訓練最後全連接層  
- 若 val 表現停滯，再解凍最後一個 block 微調（小 learning rate）

---

## 4. 訓練與評估輸出
**Student：** 作業想要圖表與指標。  
**AI：** 在 notebook 產出：  
- train/val loss & accuracy curves  
- confusion matrix  
- classification report（precision/recall/F1）  
並保存 TorchScript 模型給 Streamlit 用。

---

## 5. Streamlit Demo 設計
**Student：** Streamlit 要怎麼做？  
**AI：** 做一個上傳介面：  
- 上傳圖片 → 前處理 → 模型推論  
- 顯示 Top-3 預測與信心分數  
-（可選）Grad-CAM 顯示模型關注區域，讓 demo 更「深度學習」

---

## 6. 部署與除錯
**Student：** 部署後找不到模型檔怎麼辦？  
**AI：** Streamlit Cloud 會從 GitHub repo 取檔案，所以要把：  
- `app/model/model_resnet18.pt`  
- `app/model/class_names.json`  
一起 commit 上去；或改成啟動時自動下載（本專案先用 commit 方式簡化）。

---

## 7. 最終交付物確認
**Student：** 我需要繳：摘要、對話紀錄、GitHub + Streamlit。  
**AI：** 專案已包含：  
- `REPORT_ABSTRACT.txt`（300 字英文摘要）  
- `agent_dev_log.md`（本檔）  
- `README.md`（GitHub 說明 + 部署流程）  
- `app/streamlit_app.py`（Streamlit Demo）  
- `notebooks/train_pet_breed.ipynb`（Colab/Notebook 訓練流程）

---
