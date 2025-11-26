# SAM3 完整使用教學

## 📚 目錄
- [專案簡介](#專案簡介)
- [環境安裝](#環境安裝)
- [獲取模型權限](#獲取模型權限)
- [基礎使用](#基礎使用)
- [進階功能](#進階功能)
- [範例程式碼](#範例程式碼)
- [常見問題](#常見問題)

---

## 🎯 專案簡介

**SAM 3 (Segment Anything Model 3)** 是 Meta AI 開發的統一分割基礎模型，支援：

### 核心特性
- ✅ **開放詞彙分割** - 支援 270K+ 獨特概念
- 📸 **圖像分割** - 文字/點/框/遮罩提示
- 🎬 **視頻追蹤** - 跨幀物體追蹤
- 🤖 **Agent 整合** - 與 LLM 協同工作
- 🎯 **848M 參數** - 達到人類效能的 75-80%

### 三種主要模式

1. **PCS (Promptable Concept Segmentation)** - 概念分割
   - 使用文字描述分割所有匹配的物體
   - 例如: "穿紅色衣服的人"、"圓形物體"

2. **PVS (Promptable Visual Segmentation)** - 視覺分割
   - 類似 SAM1/SAM2 的互動式分割
   - 使用點擊、框選精確分割單一物體

3. **Agent 模式** - 智能分割助手
   - 結合 MLLM 處理複雜查詢
   - 自動分解任務並執行

---

## 🛠️ 環境安裝

### 1. 系統需求

```bash
# 硬體需求
- NVIDIA GPU (建議 16GB+ VRAM)
- CUDA 12.6 或更高版本
- 16GB+ RAM

# 軟體需求
- Python 3.12+
- PyTorch 2.7+
- Linux/macOS (Windows 需要 WSL2)
```

### 2. 創建 Conda 環境

```bash
# 創建新環境
conda create -n sam3 python=3.12
conda activate sam3

# 安裝 PyTorch (CUDA 12.6)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3. 安裝 SAM3

```bash
# 方法 1: 從 GitHub 安裝 (推薦用於研究)
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# 方法 2: 從 Hugging Face Transformers 使用 (推薦用於應用)
pip install transformers accelerate
# 不需要額外安裝,直接使用 transformers API
```

### 4. 安裝額外依賴

```bash
# 用於 Jupyter Notebook 示例
pip install -e ".[notebooks]"

# 用於訓練和開發
pip install -e ".[train,dev]"

# 或個別安裝常用套件
pip install jupyter matplotlib opencv-python pillow requests
```

---

## 🔑 獲取模型權限

### 步驟 1: 申請訪問權限

1. 訪問 [Hugging Face SAM3 頁面](https://huggingface.co/facebook/sam3)
2. 點擊 "Access repository"
3. 填寫申請表單並提交
4. 等待 Meta 團隊批准（通常幾小時內）

### 步驟 2: 設置 Hugging Face Token

```bash
# 在 Hugging Face 網站生成 access token
# 前往: https://huggingface.co/settings/tokens

# 登錄到 Hugging Face CLI
huggingface-cli login

# 或使用 Python
from huggingface_hub import login
login(token="your_token_here")
```

### 步驟 3: 驗證訪問

```python
# 測試是否能訪問模型
from transformers import Sam3Model
model = Sam3Model.from_pretrained("facebook/sam3")
print("✅ 成功載入模型!")
```

---

## 🚀 基礎使用

### 方法 A: 使用原生 SAM3 API (推薦用於圖像)

#### 圖像分割 - 文字提示

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 1. 載入模型
model = build_sam3_image_model()
processor = Sam3Processor(model)

# 2. 載入圖像
image = Image.open("your_image.jpg")
inference_state = processor.set_image(image)

# 3. 使用文字提示分割
output = processor.set_text_prompt(
    state=inference_state, 
    prompt="a red car"
)

# 4. 獲取結果
masks = output["masks"]      # 分割遮罩
boxes = output["boxes"]      # 邊界框
scores = output["scores"]    # 信心分數

print(f"找到 {len(masks)} 個物體")
```

#### 視頻追蹤 - 文字提示

```python
from sam3.model_builder import build_sam3_video_predictor

# 1. 建立視頻預測器
video_predictor = build_sam3_video_predictor()

# 2. 開始會話
video_path = "your_video.mp4"  # 或 JPEG 資料夾路徑
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]

# 3. 添加文字提示
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="person wearing red shirt",
    )
)

# 4. 在視頻中傳播分割
for frame_output in video_predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
        propagation_direction="both",
    )
):
    frame_idx = frame_output["frame_idx"]
    masks = frame_output["outputs"]["masks"]
    print(f"處理第 {frame_idx} 幀")

# 5. 關閉會話
video_predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)
```

### 方法 B: 使用 Hugging Face Transformers API (推薦)

#### 圖像分割 - 文字提示

```python
from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 載入模型和處理器
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# 2. 載入圖像
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# 3. 處理輸入
inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

# 4. 推理
with torch.no_grad():
    outputs = model(**inputs)

# 5. 後處理
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

# 6. 結果
print(f"找到 {len(results['masks'])} 個物體")
print(f"遮罩形狀: {results['masks'].shape}")
print(f"邊界框: {results['boxes'].shape}")
print(f"信心分數: {results['scores']}")
```

#### 視覺化結果

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def overlay_masks(image, masks, boxes=None):
    """在圖像上疊加分割遮罩"""
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for i, (mask, color) in enumerate(zip(masks, colors)):
        mask_img = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    
    return image

# 使用
result_image = overlay_masks(image, results["masks"])
result_image.show()
# 或儲存
result_image.save("result.png")
```

---

## 🎨 進階功能

### 1. 多種提示組合

#### 文字 + 框提示

```python
# 使用文字描述 "handle",但排除爐子把手
text = "handle"
oven_handle_box = [40, 183, 318, 204]  # [x1, y1, x2, y2]

inputs = processor(
    images=kitchen_image,
    text=text,
    input_boxes=[[oven_handle_box]],
    input_boxes_labels=[[0]],  # 0 = 負面提示 (排除)
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]
# 結果: 會分割所有把手,但不包括爐子把手
```

#### 多個框提示 (正負例)

```python
# 使用兩個正面框來定義概念
dial_box = [59, 144, 76, 163]
button_box = [87, 148, 104, 159]

inputs = processor(
    images=kitchen_image,
    input_boxes=[[dial_box, button_box]],
    input_boxes_labels=[[1, 1]],  # 兩個都是正面
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]
# 結果: 分割所有類似旋鈕和按鈕的物體
```

### 2. 批次推理

```python
# 處理多張圖像,不同提示
images = [image1, image2, image3]
text_prompts = ["cat", "dog", "car"]

inputs = processor(
    images=images, 
    text=text_prompts, 
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)

for i, result in enumerate(results):
    print(f"圖像 {i}: 找到 {len(result['masks'])} 個物體")
```

### 3. 視頻分割 (Transformers API)

```python
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator

device = Accelerator().device
model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

# 載入視頻
video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)

# 初始化會話
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

# 添加文字提示
text = "person"
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text=text,
)

# 處理所有幀
outputs_per_frame = {}
for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session, 
    max_frame_num_to_track=50
):
    processed_outputs = processor.postprocess_outputs(
        inference_session, 
        model_outputs
    )
    outputs_per_frame[model_outputs.frame_idx] = processed_outputs
    print(f"處理第 {model_outputs.frame_idx} 幀")

print(f"✅ 完成! 處理了 {len(outputs_per_frame)} 幀")
```

### 4. 互動式分割 (SAM3 Tracker)

```python
from transformers import Sam3TrackerProcessor, Sam3TrackerModel

model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

image = Image.open("truck.jpg").convert("RGB")

# 單點點擊
input_points = [[[[500, 375]]]]  # [batch, obj, points, coords]
input_labels = [[[1]]]  # 1 = 正面點擊

inputs = processor(
    images=image, 
    input_points=input_points, 
    input_labels=input_labels, 
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(
    outputs.pred_masks.cpu(), 
    inputs["original_sizes"]
)[0]

print(f"生成 {masks.shape[1]} 個遮罩候選")
```

### 5. 自動遮罩生成

```python
from transformers import pipeline

# 使用 pipeline API
generator = pipeline("mask-generation", model="facebook/sam3", device=0)

image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
outputs = generator(image_url, points_per_batch=64)

print(f"自動生成 {len(outputs['masks'])} 個遮罩")
```

---

## 📝 範例程式碼

### 完整圖像分割範例

```python
#!/usr/bin/env python3
"""
SAM3 圖像分割完整範例
支援多種提示類型和視覺化
"""

import torch
from transformers import Sam3Processor, Sam3Model
from PIL import Image, ImageDraw, ImageFont
import requests
import matplotlib.pyplot as plt
import numpy as np

def load_model(device="cuda"):
    """載入 SAM3 模型"""
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    return model, processor

def segment_with_text(model, processor, image, text_prompt, device="cuda"):
    """使用文字提示分割"""
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    return results

def visualize_results(image, results, text_prompt):
    """視覺化分割結果"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 原始圖像
    axes[0].imshow(image)
    axes[0].set_title("原始圖像")
    axes[0].axis('off')
    
    # 分割結果
    axes[1].imshow(image)
    
    # 疊加遮罩和邊界框
    masks = results['masks'].cpu().numpy()
    boxes = results['boxes'].cpu().numpy()
    scores = results['scores'].cpu().numpy()
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
    
    for i, (mask, box, score, color) in enumerate(zip(masks, boxes, scores, colors)):
        # 顯示遮罩
        axes[1].imshow(mask, alpha=0.3, cmap='jet')
        
        # 顯示邊界框
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # 顯示分數
        axes[1].text(
            x1, y1-5, f'{score:.2f}',
            color='white', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
        )
    
    axes[1].set_title(f'分割結果: "{text_prompt}" ({len(masks)} 個物體)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'sam3_result_{text_prompt.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主函數"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")
    
    # 載入模型
    print("載入 SAM3 模型...")
    model, processor = load_model(device)
    
    # 載入圖像
    print("載入圖像...")
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    # 分割不同概念
    prompts = ["ear", "nose", "eye"]
    
    for prompt in prompts:
        print(f"\n分割: {prompt}")
        results = segment_with_text(model, processor, image, prompt, device)
        print(f"找到 {len(results['masks'])} 個物體")
        visualize_results(image, results, prompt)

if __name__ == "__main__":
    main()
```

### 視頻追蹤範例

```python
#!/usr/bin/env python3
"""
SAM3 視頻追蹤範例
追蹤視頻中的特定物體
"""

import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
from accelerate import Accelerator
import cv2
import numpy as np
from pathlib import Path

def track_objects_in_video(video_path, text_prompt, output_dir="output"):
    """在視頻中追蹤物體"""
    device = Accelerator().device
    print(f"使用設備: {device}")
    
    # 載入模型
    print("載入模型...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    
    # 載入視頻
    print(f"載入視頻: {video_path}")
    video_frames, _ = load_video(video_path)
    print(f"視頻幀數: {len(video_frames)}")
    
    # 初始化會話
    print("初始化推理會話...")
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    
    # 添加文字提示
    print(f"添加文字提示: '{text_prompt}'")
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=text_prompt,
    )
    
    # 處理視頻
    print("開始追蹤...")
    outputs_per_frame = {}
    
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session
    ):
        processed_outputs = processor.postprocess_outputs(
            inference_session, 
            model_outputs
        )
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs
        
        if (model_outputs.frame_idx + 1) % 30 == 0:
            print(f"已處理 {model_outputs.frame_idx + 1} 幀...")
    
    print(f"✅ 完成! 處理了 {len(outputs_per_frame)} 幀")
    
    # 保存結果
    save_video_with_masks(video_frames, outputs_per_frame, output_dir, text_prompt)
    
    return outputs_per_frame

def save_video_with_masks(video_frames, outputs_per_frame, output_dir, text_prompt):
    """將帶有遮罩的視頻保存"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"tracked_{text_prompt.replace(' ', '_')}.mp4"
    
    # 獲取視頻尺寸
    height, width = video_frames[0].shape[:2]
    
    # 創建視頻寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
    
    print(f"保存視頻到: {output_path}")
    
    for frame_idx, frame in enumerate(video_frames):
        if frame_idx in outputs_per_frame:
            outputs = outputs_per_frame[frame_idx]
            
            # 轉換為 BGR
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            
            # 疊加遮罩
            if len(outputs['masks']) > 0:
                masks = outputs['masks'].cpu().numpy()
                boxes = outputs['boxes'].cpu().numpy()
                
                for mask, box in zip(masks, boxes):
                    # 創建彩色遮罩
                    color = (0, 255, 0)  # 綠色
                    colored_mask = np.zeros_like(frame_bgr)
                    colored_mask[mask > 0.5] = color
                    
                    # 疊加
                    frame_bgr = cv2.addWeighted(frame_bgr, 1, colored_mask, 0.3, 0)
                    
                    # 繪製邊界框
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            
            out.write(frame_bgr)
        else:
            out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"✅ 視頻已保存")

def main():
    """主函數"""
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    text_prompt = "person"
    
    outputs = track_objects_in_video(video_url, text_prompt)
    
    # 統計信息
    total_detections = sum(len(out['object_ids']) for out in outputs.values())
    print(f"\n統計:")
    print(f"  總幀數: {len(outputs)}")
    print(f"  總檢測數: {total_detections}")
    print(f"  平均每幀: {total_detections / len(outputs):.1f} 個物體")

if __name__ == "__main__":
    main()
```

---

## ❓ 常見問題

### Q1: 如何處理 CUDA out of memory 錯誤?

```python
# 解決方案 1: 降低批次大小
inputs = processor(images=image, text=prompt, return_tensors="pt")
# 一次處理一張

# 解決方案 2: 使用較小的圖像
image = image.resize((800, 600))

# 解決方案 3: 使用 bfloat16
model = model.to(dtype=torch.bfloat16)

# 解決方案 4: 啟用梯度檢查點 (訓練時)
model.gradient_checkpointing_enable()
```

### Q2: 如何提高分割準確度?

```python
# 方法 1: 調整閾值
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.3,  # 降低閾值獲得更多檢測
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

# 方法 2: 使用更具體的文字提示
# ❌ 不好: "object"
# ✅ 好: "red sports car with white stripes"

# 方法 3: 組合多種提示
inputs = processor(
    images=image,
    text="car",
    input_boxes=[[approximate_box]],  # 添加大致位置
    input_boxes_labels=[[1]],
    return_tensors="pt"
)
```

### Q3: 如何處理大型視頻?

```python
# 使用串流模式
inference_session = processor.init_video_session(
    inference_device=device,
    processing_device="cpu",  # CPU 處理以節省 GPU 記憶體
    video_storage_device="cpu",  # 將幀存在 CPU
    dtype=torch.bfloat16,
)

# 逐幀處理
for frame_idx, frame in enumerate(video_frames):
    inputs = processor(images=frame, device=device, return_tensors="pt")
    
    model_outputs = model(
        inference_session=inference_session,
        frame=inputs.pixel_values[0],
        reverse=False,
    )
    
    # 立即處理並釋放記憶體
    processed = processor.postprocess_outputs(
        inference_session,
        model_outputs,
        original_sizes=inputs.original_sizes,
    )
    
    # 保存或處理結果
    save_frame_result(processed)
    
    # 清理
    del inputs, model_outputs, processed
    torch.cuda.empty_cache()
```

### Q4: 支援哪些語言的文字提示?

SAM3 主要在英文數據上訓練,但也支援其他語言:

```python
# 英文 (最佳)
text = "a red car"

# 中文 (部分支援)
text = "一輛紅色的汽車"

# 其他語言
text = "ein rotes Auto"  # 德文

# 建議: 使用簡單、描述性的英文以獲得最佳結果
```

### Q5: 如何在生產環境中部署?

```python
# 優化建議

# 1. 使用 TorchScript
model = torch.jit.script(model)

# 2. 使用 ONNX (如果支援)
torch.onnx.export(model, ...)

# 3. 使用 FastAPI 建立 API
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.post("/segment/")
async def segment_image(file: UploadFile, text_prompt: str):
    image = Image.open(file.file)
    results = segment_with_text(model, processor, image, text_prompt)
    return {"masks": results["masks"].tolist(), "boxes": results["boxes"].tolist()}

# 4. 使用 Docker 容器化
# Dockerfile 範例
"""
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04
RUN pip install torch transformers
COPY . /app
CMD ["python", "server.py"]
"""

# 5. 批次處理優化
def batch_inference(images, prompts, batch_size=4):
    results = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        batch_results = model_inference(batch_images, batch_prompts)
        results.extend(batch_results)
    return results
```

### Q6: 如何微調 SAM3 在自訂數據上?

```bash
# 參考訓練文檔
cd sam3
pip install -e ".[train]"

# 使用提供的配置文件
python sam3/train/train.py -c configs/your_config.yaml

# 自訂數據集格式 (COCO 格式)
"""
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [...],
      "bbox": [x, y, w, h],
      "area": ...,
      "iscrowd": 0
    }
  ],
  "categories": [...]
}
"""
```

---

## 📚 更多資源

### 官方資源
- 📄 [論文](https://arxiv.org/abs/2511.16719)
- 🌐 [專案頁面](https://ai.meta.com/sam3)
- 💻 [GitHub Repo](https://github.com/facebookresearch/sam3)
- 🤗 [Hugging Face Model](https://huggingface.co/facebook/sam3)
- 📝 [Blog 文章](https://ai.meta.com/blog/segment-anything-model-3/)

### Jupyter Notebook 範例
- `sam3_image_predictor_example.ipynb` - 圖像分割
- `sam3_video_predictor_example.ipynb` - 視頻追蹤
- `sam3_image_batched_inference.ipynb` - 批次推理
- `sam3_agent.ipynb` - Agent 模式
- `sam3_for_sam1_task_example.ipynb` - SAM1 任務
- `sam3_for_sam2_video_task_example.ipynb` - SAM2 任務

### 社群資源
- 🎨 [Hugging Face Spaces](https://huggingface.co/spaces/akhaliq/sam3) - 線上 Demo
- 🐍 [Python 範例](https://github.com/facebookresearch/sam3/tree/main/examples)
- 📊 [評估腳本](https://github.com/facebookresearch/sam3/tree/main/scripts/eval)

---

## 🎯 快速開始檢查清單

- [ ] ✅ 安裝 Python 3.12+ 和 PyTorch 2.7+
- [ ] ✅ 從 GitHub 或 pip 安裝 SAM3
- [ ] ✅ 申請並獲得 Hugging Face 模型訪問權限
- [ ] ✅ 設置 Hugging Face Token
- [ ] ✅ 運行第一個圖像分割範例
- [ ] ✅ 嘗試不同的提示類型
- [ ] ✅ 探索視頻追蹤功能
- [ ] ✅ 查看 Jupyter Notebook 範例

---

## 📧 支援

如有問題:
1. 查看 [GitHub Issues](https://github.com/facebookresearch/sam3/issues)
2. 參考 [Hugging Face Discussions](https://huggingface.co/facebook/sam3/discussions)
3. 閱讀官方文檔和範例

---

**祝你使用 SAM3 愉快! 🎉**

最後更新: 2025-11-25
