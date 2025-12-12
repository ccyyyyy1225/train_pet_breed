# 寵物品種辨識（Pet Breed Classification）— Transfer Learning + Streamlit Demo

本專案示範如何使用 **預訓練 CNN（ResNet18）** 進行 **小樣本寵物品種辨識**，並延伸成 **Streamlit Web App**：
- 資料夾每類只有 5 張也能做（以「資料切分 + 強資料增強 + Transfer Learning」完成可展示的 Demo）
- 產出：Learning Curves、Confusion Matrix、Classification Report、Top-K 預測、（可選）Grad-CAM 可視化
- 部署：GitHub + Streamlit Cloud（streamlit.app）

---

## 1) 你目前的資料格式（OK）
請把資料放成：
```
dataset/
  American Shorthair/   (5 images)
  British Shorthair/    (5 images)
  Golden Retriever/     (5 images)
  Husky/                (5 images)
  Labrador/             (5 images)
  Maine Coon/           (5 images)
  Ragdoll/              (5 images)
  Shiba Inu/            (5 images)
```

本專案會自動切成：
- train：3 張/類
- val：1 張/類
- test：1 張/類

---

## 2) Colab / Notebook 訓練（建議）
請開啟 `notebooks/train_pet_breed.ipynb`，照 cell 順序執行。完成後會輸出：
- `exports/model_resnet18.pt`（TorchScript，可直接給 Streamlit 用）
- `exports/class_names.json`

接著把這兩個檔案複製到：
```
app/model/model_resnet18.pt
app/model/class_names.json
```

---

## 3) 本機測試 Streamlit
在專案根目錄執行：
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 4) 部署到 Streamlit Cloud（streamlit.app）
1. 把整個專案 push 到 GitHub
2. 到 Streamlit Cloud 連 GitHub repo，設定：
   - Main file path：`app/streamlit_app.py`
3. 若部署時沒找到模型檔，請確認 repo 內有：
   - `app/model/model_resnet18.pt`
   - `app/model/class_names.json`

---

## 5) 300 字英文摘要（可交作業）
見 `REPORT_ABSTRACT.txt`

---

## 6) Agent 開發過程對話紀錄
見 `agent_dev_log.md`

---

## 7) 常見問題
### Q1：每類只有 5 張會不會太少？
會偏少，所以這個專案採用：
- 預訓練 ResNet18（ImageNet）
- 強資料增強（隨機裁切、翻轉、亮度對比、輕微旋轉）
- 凍結 backbone、只訓練分類 head（再視情況微調最後一個 block）
來做 **可展示的「小樣本 demo」**。正式研究要更多資料。

### Q2：資料夾名稱有空白（例如 Shiba Inu）會有問題嗎？
不會。程式會以資料夾名稱作為類別名稱；Streamlit 顯示時也用原名稱。

---

## 8) 你可以在報告中寫的延伸亮點（建議）
- Top-3 預測與信心分數可視化
- Grad-CAM：模型到底在看耳朵/臉/毛色哪裡判斷
- 小樣本限制與資料增強策略的效果比較
