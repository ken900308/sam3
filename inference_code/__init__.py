# SAM3 Inference Code
# ===================
# 
# 這個資料夾包含所有 SAM3 推論相關的程式碼
# 
# 檔案結構:
# ├── image_inference.py      - 圖片分割推論
# ├── video_inference.py      - 視訊分割推論 (離線處理)
# ├── streaming_inference.py  - 即時串流推論 (Webcam/視訊)
# └── outputs/                - 輸出結果目錄
#
# 使用方式請參考各檔案的說明文檔

import os

# 設定輸出目錄
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
