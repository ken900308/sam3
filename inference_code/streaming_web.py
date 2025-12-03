#!/usr/bin/env python3
"""
SAM3 網頁串流推論腳本
====================
透過瀏覽器即時觀看 SAM3 分割結果

使用方法:
    python streaming_web.py --source webcam --prompt "person"
    python streaming_web.py --source 0 --prompt "hand" --port 8080

然後開啟瀏覽器訪問:
    http://localhost:5000

功能:
    - 即時 MJPEG 串流顯示
    - 網頁介面調整參數
    - 支援 webcam / 視訊檔案 / RTSP
"""

import argparse
import os
import sys
import time
import threading
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from flask import Flask, Response, render_template_string, request, jsonify

# 確保可以 import sam3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 設定 TF32 以提升 Ampere GPU 效能
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Flask app
app = Flask(__name__)

# 全域變數
frame_buffer = None
frame_lock = threading.Lock()
current_prompt = "person"
current_fps = 0.0
current_objects = 0
is_running = True
confidence_threshold = 0.5
show_boxes = False  # 控制是否顯示邊界框


def overlay_masks(frame, masks, boxes, scores, alpha=0.5, show_boxes=False):
    """在影格上疊加遮罩"""
    if masks is None or len(masks) == 0:
        return frame
    
    overlay = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    
    n_masks = len(masks)
    cmap = plt.colormaps.get_cmap('rainbow')
    colors = [
        tuple(int(c * 255) for c in cmap(i / max(n_masks, 1))[:3])
        for i in range(n_masks)
    ]
    
    for i, (mask, color) in enumerate(zip(masks, colors)):
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        
        if mask_np.size == 0:
            continue
            
        if mask_np.shape[0] != frame_h or mask_np.shape[1] != frame_w:
            mask_np = cv2.resize(
                mask_np.astype(np.uint8), 
                (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST
            )
        
        mask_bool = mask_np.astype(bool)
        color_bgr = color[::-1]
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + 
            np.array(color_bgr) * alpha
        ).astype(np.uint8)
        
        # 只在 show_boxes=True 時繪製邊界框
        if show_boxes and boxes is not None and i < len(boxes):
            box = boxes[i]
            if hasattr(box, 'cpu'):
                box = box.cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 2)
            
            if scores is not None and i < len(scores):
                score = float(scores[i])
                label = f"{score:.2f}"
                cv2.putText(
                    overlay, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2
                )
    
    return overlay


def inference_thread(source, device):
    """推論執行緒"""
    global frame_buffer, current_fps, current_objects, is_running, current_prompt, confidence_threshold, show_boxes
    
    # 開啟視訊來源
    if source == "webcam" or source == "0":
        cap = cv2.VideoCapture(0)
    elif source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ 無法開啟視訊來源: {source}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 載入模型
    print("🔧 載入模型...")
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    
    model = build_sam3_image_model(bpe_path=bpe_path, device=device)
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    
    print("✅ 模型載入完成，開始推論...")
    
    fps_queue = deque(maxlen=30)
    last_prompt = current_prompt
    
    while is_running:
        ret, frame = cap.read()
        if not ret:
            continue
        
        start_time = time.time()
        
        # 檢查 prompt 是否改變
        if current_prompt != last_prompt:
            last_prompt = current_prompt
            print(f"📝 提示詞已更改為: {current_prompt}")
        
        # 更新信心閾值
        processor.confidence_threshold = confidence_threshold
        
        # 轉換為 PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # 推論
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = processor.set_image(pil_image)
            inference_state = processor.set_text_prompt(
                state=inference_state, 
                prompt=current_prompt
            )
        
        masks = inference_state.get("masks")
        boxes = inference_state.get("boxes")
        scores = inference_state.get("scores")
        
        # 疊加遮罩 (根據 show_boxes 決定是否顯示邊界框)
        result_frame = overlay_masks(frame, masks, boxes, scores, show_boxes=show_boxes)
        
        # 計算 FPS
        elapsed = time.time() - start_time
        fps_queue.append(1.0 / elapsed if elapsed > 0 else 0)
        current_fps = sum(fps_queue) / len(fps_queue)
        current_objects = len(masks) if masks is not None else 0
        
        # 顯示資訊
        info_text = f"FPS: {current_fps:.1f} | Objects: {current_objects} | Prompt: {current_prompt}"
        cv2.putText(
            result_frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # 更新 buffer
        with frame_lock:
            frame_buffer = result_frame.copy()
    
    cap.release()
    print("推論執行緒已停止")


def generate_frames():
    """產生 MJPEG 串流"""
    global frame_buffer
    
    while is_running:
        frame = None
        
        with frame_lock:
            if frame_buffer is not None:
                frame = frame_buffer.copy()
        
        if frame is None:
            time.sleep(0.1)
            continue
        
        # 編碼為 JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS 串流


# HTML 模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>SAM3 即時串流</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        h1 {
            color: #00d4ff;
            text-align: center;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
        }
        .video-container img {
            max-width: 100%;
            border: 3px solid #00d4ff;
            border-radius: 10px;
        }
        .controls {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .control-group {
            margin: 15px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        label {
            width: 120px;
            font-weight: bold;
        }
        input[type="text"] {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
        }
        input[type="range"] {
            flex: 1;
        }
        button {
            background: #00d4ff;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background: #00a8cc;
        }
        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
        }
        .stat {
            background: #16213e;
            padding: 15px 25px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00d4ff;
        }
        .stat-label {
            font-size: 14px;
            color: #888;
        }
        .examples {
            margin-top: 10px;
            color: #888;
            font-size: 14px;
        }
        .examples span {
            background: #0f3460;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 0 3px;
            cursor: pointer;
        }
        .examples span:hover {
            background: #00d4ff;
            color: #1a1a2e;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 SAM3 即時串流推論</h1>
        
        <div class="video-container">
            <img src="/video_feed" alt="SAM3 Stream">
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="fps">0.0</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="objects">0</div>
                <div class="stat-label">偵測物件</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="prompt-display">-</div>
                <div class="stat-label">當前提示詞</div>
            </div>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>提示詞：</label>
                <input type="text" id="prompt" placeholder="輸入要偵測的物件..." value="{{ prompt }}">
                <button onclick="updatePrompt()">更新</button>
            </div>
            <div class="examples">
                範例：
                <span onclick="setPrompt('person')">person</span>
                <span onclick="setPrompt('hand')">hand</span>
                <span onclick="setPrompt('face')">face</span>
                <span onclick="setPrompt('cat')">cat</span>
                <span onclick="setPrompt('dog')">dog</span>
                <span onclick="setPrompt('cup')">cup</span>
                <span onclick="setPrompt('phone')">phone</span>
                <span onclick="setPrompt('keyboard')">keyboard</span>
            </div>
            
            <div class="control-group">
                <label>信心閾值：</label>
                <input type="range" id="confidence" min="0.1" max="0.9" step="0.1" value="0.5">
                <span id="conf-value">0.5</span>
            </div>
            
            <div class="control-group">
                <label>邊界框：</label>
                <button id="toggle-boxes" onclick="toggleBoxes()">關閉</button>
                <span style="color: #888; font-size: 14px;">（點擊切換顯示/隱藏邊界框）</span>
            </div>
        </div>
    </div>
    
    <script>
        function updatePrompt() {
            const prompt = document.getElementById('prompt').value;
            fetch('/set_prompt', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt})
            });
        }
        
        function setPrompt(text) {
            document.getElementById('prompt').value = text;
            updatePrompt();
        }
        
        function toggleBoxes() {
            fetch('/toggle_boxes', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                const btn = document.getElementById('toggle-boxes');
                btn.textContent = data.show_boxes ? '開啟' : '關閉';
                btn.style.background = data.show_boxes ? '#00d4ff' : '#666';
            });
        }
        
        document.getElementById('confidence').addEventListener('change', function() {
            const value = this.value;
            document.getElementById('conf-value').textContent = value;
            fetch('/set_confidence', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({confidence: parseFloat(value)})
            });
        });
        
        // 更新統計資訊
        setInterval(function() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('objects').textContent = data.objects;
                    document.getElementById('prompt-display').textContent = data.prompt;
                    // 更新邊界框按鈕狀態
                    const btn = document.getElementById('toggle-boxes');
                    btn.textContent = data.show_boxes ? '開啟' : '關閉';
                    btn.style.background = data.show_boxes ? '#00d4ff' : '#666';
                });
        }, 500);
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, prompt=current_prompt)


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stats')
def stats():
    return jsonify({
        'fps': current_fps,
        'objects': current_objects,
        'prompt': current_prompt,
        'show_boxes': show_boxes
    })


@app.route('/set_prompt', methods=['POST'])
def set_prompt():
    global current_prompt
    data = request.get_json()
    current_prompt = data.get('prompt', 'person')
    print(f"📝 提示詞更新: {current_prompt}")
    return jsonify({'success': True, 'prompt': current_prompt})


@app.route('/toggle_boxes', methods=['POST'])
def toggle_boxes():
    global show_boxes
    show_boxes = not show_boxes
    print(f"📦 邊界框: {'開啟' if show_boxes else '關閉'}")
    return jsonify({'success': True, 'show_boxes': show_boxes})


@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    global confidence_threshold
    data = request.get_json()
    confidence_threshold = data.get('confidence', 0.5)
    print(f"🎚️ 信心閾值更新: {confidence_threshold}")
    return jsonify({'success': True, 'confidence': confidence_threshold})


def main():
    global current_prompt, is_running
    
    parser = argparse.ArgumentParser(description="SAM3 網頁串流推論")
    parser.add_argument("--source", type=str, default="webcam", help="視訊來源")
    parser.add_argument("--prompt", type=str, default="person", help="初始提示詞")
    parser.add_argument("--port", type=int, default=5000, help="網頁伺服器埠號")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="綁定的主機位址")
    args = parser.parse_args()
    
    current_prompt = args.prompt
    
    print("=" * 60)
    print("SAM3 網頁串流推論")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  使用 CPU (速度會很慢)")
    
    print(f"\n📹 視訊來源: {args.source}")
    print(f"📝 初始提示詞: {args.prompt}")
    print(f"\n🌐 開啟瀏覽器訪問: http://localhost:{args.port}")
    print("   按 Ctrl+C 停止伺服器")
    print("=" * 60)
    
    # 啟動推論執行緒
    inference = threading.Thread(target=inference_thread, args=(args.source, device))
    inference.daemon = True
    inference.start()
    
    # 等待模型載入
    time.sleep(2)
    
    try:
        # 啟動 Flask 伺服器
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n⏹️ 停止伺服器...")
    finally:
        is_running = False


if __name__ == "__main__":
    main()
