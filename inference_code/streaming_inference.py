#!/usr/bin/env python3
"""
SAM3 即時串流推論腳本 (Webcam / 視訊檔案 / RTSP)
================================================
基於官方 examples 修改，支援即時攝影機串流 detection

使用方法:
    python streaming_inference.py --source webcam --prompt "person"
    python streaming_inference.py --source 0 --prompt "face"
    python streaming_inference.py --source video.mp4 --prompt "car"
    python streaming_inference.py --source rtsp://... --prompt "person"

注意:
    - 預期速度: 3-4 FPS (RTX 4060 Ti)
    - 需要 X11 顯示支援 (Docker 中使用 ./run_docker.sh shell)
    - 按 'q' 退出
    - 按 's' 截圖
    - 按 'p' 暫停/繼續
"""

import argparse
import os
import sys
import time
from collections import deque

# 設定 OpenCV 環境變數 (必須在 import cv2 之前)
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_X11_NO_MITSHM'] = '1'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 確保可以 import sam3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 設定輸出目錄
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 設定 TF32 以提升 Ampere GPU 效能
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 即時串流推論")
    parser.add_argument(
        "--source", 
        type=str, 
        default="webcam",
        help="輸入來源: 'webcam', 視訊檔案路徑, 或攝影機編號 (0, 1, ...)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="person",
        help="文字提示 (要分割的物件)"
    )
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.5,
        help="信心閾值 (0-1)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="處理寬度 (較小 = 較快)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="處理高度 (較小 = 較快)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="不顯示視窗 (僅儲存結果)"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="儲存輸出視訊"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最大處理幀數 (用於測試)"
    )
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        default=False,
        help="顯示邊界框 (預設關閉)"
    )
    return parser.parse_args()


def overlay_masks(frame, masks, boxes, scores, alpha=0.5, show_boxes=False):
    """在影格上疊加遮罩
    
    Args:
        frame: 原始影格
        masks: 分割遮罩
        boxes: 邊界框
        scores: 信心分數
        alpha: 遮罩透明度
        show_boxes: 是否顯示邊界框 (True=顯示, False=隱藏)
    """
    if masks is None or len(masks) == 0:
        return frame
    
    overlay = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    
    # 生成顏色
    n_masks = len(masks)
    cmap = plt.colormaps.get_cmap('rainbow')
    colors = [
        tuple(int(c * 255) for c in cmap(i / max(n_masks, 1))[:3])
        for i in range(n_masks)
    ]
    
    for i, (mask, color) in enumerate(zip(masks, colors)):
        # 轉換遮罩
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        # 確保 mask 是 2D
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        
        # 跳過空的 mask
        if mask_np.size == 0:
            continue
            
        # 調整遮罩大小到影格尺寸
        if mask_np.shape[0] != frame_h or mask_np.shape[1] != frame_w:
            mask_np = cv2.resize(
                mask_np.astype(np.uint8), 
                (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST
            )
        
        mask_bool = mask_np.astype(bool)
        
        # 疊加顏色 (BGR)
        color_bgr = color[::-1]
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + 
            np.array(color_bgr) * alpha
        ).astype(np.uint8)
        
        # 繪製邊界框 (只在 show_boxes=True 時顯示)
        if show_boxes and boxes is not None and i < len(boxes):
            box = boxes[i]
            if hasattr(box, 'cpu'):
                box = box.cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 2)
            
            # 顯示分數
            if scores is not None and i < len(scores):
                score = float(scores[i])
                label = f"{score:.2f}"
                cv2.putText(
                    overlay, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2
                )
    
    return overlay


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SAM3 即時串流推論")
    print("=" * 60)
    
    # 檢查 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  使用 CPU (速度會很慢)")
    
    # 開啟視訊來源
    if args.source == "webcam" or args.source == "0":
        cap = cv2.VideoCapture(0)
        source_name = "webcam"
    elif args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        source_name = f"camera_{args.source}"
    else:
        cap = cv2.VideoCapture(args.source)
        source_name = os.path.splitext(os.path.basename(args.source))[0]
    
    if not cap.isOpened():
        print(f"❌ 無法開啟視訊來源: {args.source}")
        sys.exit(1)
    
    # 設定解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    print(f"\n📹 視訊來源: {args.source}")
    print(f"   解析度: {actual_width}x{actual_height}")
    print(f"   FPS: {fps:.1f}")
    print(f"   提示: '{args.prompt}'")
    
    # 測試 X11 顯示
    if not args.no_display:
        try:
            # 讀取一幀來測試
            ret, test_frame = cap.read()
            if ret:
                cv2.namedWindow("SAM3 Streaming", cv2.WINDOW_NORMAL)
                cv2.imshow("SAM3 Streaming", test_frame)
                cv2.waitKey(1)
                print("✅ X11 顯示正常")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到開頭
        except Exception as e:
            print(f"⚠️  X11 顯示有問題: {e}")
            print("   建議使用 --no-display --save-video 模式")
            args.no_display = True
    
    # 載入模型
    print("\n🔧 載入模型...")
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    
    model = build_sam3_image_model(bpe_path=bpe_path, device=device)
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    
    # 視訊寫入器
    # 注意：使用實際推論 FPS 而不是攝影機 FPS，避免播放加速
    # SAM3 約 3-4 FPS，所以錄製用這個速度
    output_fps = 4.0  # 預估的推論 FPS
    video_writer = None
    if args.save_video:
        output_path = os.path.join(
            OUTPUT_DIR, 
            f"{source_name}_{args.prompt.replace(' ', '_')}_stream.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path, fourcc, output_fps,  # 使用實際推論速度
            (actual_width, actual_height)
        )
        print(f"💾 錄製到: {output_path}")
        print(f"   輸出 FPS: {output_fps} (依實際推論速度)")
    
    # FPS 計算
    fps_queue = deque(maxlen=30)
    frame_count = 0
    screenshot_count = 0
    paused = False
    last_result_frame = None
    
    print("\n🎬 開始串流推論...")
    print("   按 'q' 退出")
    print("   按 's' 截圖")
    print("   按 'p' 暫停/繼續")
    print("   按 'c' 更換提示詞")
    print("-" * 60)
    
    try:
        while True:
            # 處理暫停
            if paused:
                if last_result_frame is not None:
                    # 顯示暫停狀態
                    display_frame = last_result_frame.copy()
                    cv2.putText(
                        display_frame, "PAUSED - Press 'p' to resume", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )
                    cv2.imshow("SAM3 Streaming", display_frame)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p'):
                    paused = False
                    print("\n▶️  繼續推論")
                elif key == ord('q'):
                    print("\n⏹️  使用者中斷")
                    break
                elif key == ord('s') and last_result_frame is not None:
                    screenshot_count += 1
                    screenshot_path = os.path.join(
                        OUTPUT_DIR,
                        f"{source_name}_screenshot_{screenshot_count:03d}.png"
                    )
                    cv2.imwrite(screenshot_path, last_result_frame)
                    print(f"\n📸 截圖已儲存: {screenshot_path}")
                continue
            
            ret, frame = cap.read()
            if not ret:
                if args.source != "webcam" and not args.source.isdigit():
                    print("\n📹 視訊結束")
                    break
                continue
            
            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                break
            
            start_time = time.time()
            
            # 轉換為 PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 推論
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = processor.set_image(pil_image)
                inference_state = processor.set_text_prompt(
                    state=inference_state, 
                    prompt=args.prompt
                )
            
            # 取得結果
            masks = inference_state.get("masks")
            boxes = inference_state.get("boxes")
            scores = inference_state.get("scores")
            
            # 疊加遮罩 (根據 --show-boxes 決定是否顯示邊界框)
            result_frame = overlay_masks(frame, masks, boxes, scores, show_boxes=args.show_boxes)
            
            # 計算 FPS
            elapsed = time.time() - start_time
            fps_queue.append(1.0 / elapsed if elapsed > 0 else 0)
            avg_fps = sum(fps_queue) / len(fps_queue)
            
            # 顯示資訊
            num_objects = len(masks) if masks is not None else 0
            info_text = f"FPS: {avg_fps:.1f} | Objects: {num_objects} | Prompt: {args.prompt}"
            cv2.putText(
                result_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            # 儲存最後一幀 (用於暫停時顯示)
            last_result_frame = result_frame.copy()
            
            # 儲存視訊
            if video_writer:
                video_writer.write(result_frame)
            
            # 顯示
            if not args.no_display:
                cv2.imshow("SAM3 Streaming", result_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  使用者中斷")
                    break
                elif key == ord('s'):
                    screenshot_count += 1
                    screenshot_path = os.path.join(
                        OUTPUT_DIR,
                        f"{source_name}_screenshot_{screenshot_count:03d}.png"
                    )
                    cv2.imwrite(screenshot_path, result_frame)
                    print(f"\n📸 截圖已儲存: {screenshot_path}")
                elif key == ord('p'):
                    paused = True
                    print("\n⏸️  暫停推論")
            
            # 顯示進度
            if frame_count % 10 == 0:
                print(f"\r   Frame {frame_count} | FPS: {avg_fps:.1f} | Objects: {num_objects}    ", end="", flush=True)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Ctrl+C 中斷")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"處理完成！共 {frame_count} 幀")
    if video_writer:
        print(f"視訊已儲存到: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
