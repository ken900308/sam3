#!/usr/bin/env python3
"""
SAM3 GUI 串流推論腳本 (使用 Pygame)
====================================
使用 Pygame 作為 GUI 後端，避免 Docker X11 + OpenCV Qt5 的相容性問題

使用方法:
    python streaming_gui.py --source webcam --prompt "person"
    python streaming_gui.py --source 0 --prompt "hand"

控制:
    - 按 'q' 或 ESC 退出
    - 按 's' 截圖
    - 按 'p' 暫停/繼續
    - 按 1-8 快速切換提示詞
    - 輸入文字更改提示詞，按 Enter 確認

重要: 必須先初始化 Pygame 再載入 PyTorch，否則會 Segmentation fault
"""

import argparse
import os
import sys
import time
from collections import deque

# 環境變數必須在 import 之前設定
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['SDL_AUDIODRIVER'] = 'dummy'  # 禁用音訊避免 ALSA 錯誤
os.environ['SDL_VIDEO_X11_FORCE_EGL'] = '0'

# !! 重要：先 import 並初始化 pygame，再 import torch !!
import pygame
from pygame.locals import *
pygame.init()

# 現在可以安全 import 其他模組
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非互動後端
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

# 預設提示詞列表
QUICK_PROMPTS = [
    "person", "hand", "face", "body",
    "cat", "dog", "cup", "phone"
]


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 GUI 串流推論 (Pygame)")
    parser.add_argument("--source", type=str, default="webcam", help="視訊來源")
    parser.add_argument("--prompt", type=str, default="person", help="初始提示詞")
    parser.add_argument("--confidence", type=float, default=0.5, help="信心閾值")
    parser.add_argument("--width", type=int, default=800, help="視窗寬度")
    parser.add_argument("--height", type=int, default=600, help="視窗高度")
    return parser.parse_args()


def overlay_masks(frame, masks, boxes, scores, alpha=0.5):
    """在影格上疊加遮罩"""
    if masks is None:
        return frame
    
    # 處理空的 tensor
    try:
        if len(masks) == 0:
            return frame
    except:
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
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + 
            np.array(color) * alpha
        ).astype(np.uint8)
        
        if boxes is not None and i < len(boxes):
            box = boxes[i]
            if hasattr(box, 'cpu'):
                box = box.cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            if scores is not None and i < len(scores):
                score = float(scores[i])
                label = f"{score:.2f}"
                cv2.putText(
                    overlay, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
    
    return overlay


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SAM3 GUI 串流推論 (Pygame)")
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
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 視訊來源: {args.source}")
    print(f"   解析度: {actual_width}x{actual_height}")
    
    # 載入模型
    print("\n🔧 載入模型...")
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    
    model = build_sam3_image_model(bpe_path=bpe_path, device=device)
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    print("✅ 模型載入完成")
    
    # Pygame 已在模組載入時初始化
    pygame.display.set_caption("SAM3 即時串流推論")
    
    # 設定視窗大小
    window_width = args.width
    window_height = args.height
    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    
    # 字型
    font_large = pygame.font.SysFont('DejaVu Sans', 24)
    font_medium = pygame.font.SysFont('DejaVu Sans', 18)
    font_small = pygame.font.SysFont('DejaVu Sans', 14)
    
    # 狀態變數
    current_prompt = args.prompt
    input_text = ""
    input_active = False
    paused = False
    fps_queue = deque(maxlen=30)
    frame_count = 0
    screenshot_count = 0
    last_frame = None
    current_fps = 0.0
    current_objects = 0
    
    # 顏色
    COLOR_BG = (26, 26, 46)
    COLOR_PANEL = (22, 33, 62)
    COLOR_ACCENT = (0, 212, 255)
    COLOR_TEXT = (238, 238, 238)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (255, 0, 0)
    
    clock = pygame.time.Clock()
    running = True
    
    print("\n🎬 開始串流推論...")
    print("   按 'q' 或 ESC 退出")
    print("   按 's' 截圖")
    print("   按 'p' 暫停/繼續")
    print("   按 1-8 快速切換提示詞")
    print("   點擊輸入框或按 Tab 輸入自定義提示詞")
    print("-" * 60)
    
    # 計算佈局
    video_area_height = window_height - 120
    panel_height = 120
    
    while running:
        # 事件處理
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            
            elif event.type == VIDEORESIZE:
                window_width, window_height = event.size
                screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
                video_area_height = window_height - 120
            
            elif event.type == KEYDOWN:
                if input_active:
                    if event.key == K_RETURN:
                        if input_text.strip():
                            current_prompt = input_text.strip()
                            print(f"📝 提示詞更新: {current_prompt}")
                        input_text = ""
                        input_active = False
                    elif event.key == K_ESCAPE:
                        input_text = ""
                        input_active = False
                    elif event.key == K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode
                else:
                    if event.key == K_q or event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_s:
                        if last_frame is not None:
                            screenshot_count += 1
                            path = os.path.join(OUTPUT_DIR, f"{source_name}_screenshot_{screenshot_count:03d}.png")
                            cv2.imwrite(path, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
                            print(f"📸 截圖已儲存: {path}")
                    elif event.key == K_p:
                        paused = not paused
                        print("⏸️  暫停" if paused else "▶️  繼續")
                    elif event.key == K_TAB:
                        input_active = True
                    elif event.key in [K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8]:
                        idx = event.key - K_1
                        if idx < len(QUICK_PROMPTS):
                            current_prompt = QUICK_PROMPTS[idx]
                            print(f"📝 提示詞更新: {current_prompt}")
            
            elif event.type == MOUSEBUTTONDOWN:
                # 檢查是否點擊輸入框區域
                input_box_rect = pygame.Rect(10, window_height - 45, 300, 35)
                if input_box_rect.collidepoint(event.pos):
                    input_active = True
                else:
                    input_active = False
                    
                # 檢查快速按鈕
                btn_start_x = 320
                for i, prompt in enumerate(QUICK_PROMPTS[:4]):
                    btn_rect = pygame.Rect(btn_start_x + i * 80, window_height - 110, 75, 25)
                    if btn_rect.collidepoint(event.pos):
                        current_prompt = prompt
                        print(f"📝 提示詞更新: {current_prompt}")
                for i, prompt in enumerate(QUICK_PROMPTS[4:8]):
                    btn_rect = pygame.Rect(btn_start_x + i * 80, window_height - 80, 75, 25)
                    if btn_rect.collidepoint(event.pos):
                        current_prompt = prompt
                        print(f"📝 提示詞更新: {current_prompt}")
        
        # 清除畫面
        screen.fill(COLOR_BG)
        
        if not paused:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                start_time = time.time()
                
                # 轉換為 RGB
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
                
                # 疊加遮罩
                result_frame = overlay_masks(frame_rgb, masks, boxes, scores)
                last_frame = result_frame
                
                # 計算 FPS
                elapsed = time.time() - start_time
                fps_queue.append(1.0 / elapsed if elapsed > 0 else 0)
                current_fps = sum(fps_queue) / len(fps_queue)
                current_objects = len(masks) if masks is not None else 0
        
        # 顯示影像
        if last_frame is not None:
            # 調整大小以適應視窗
            frame_surface = pygame.surfarray.make_surface(
                np.transpose(last_frame, (1, 0, 2))
            )
            
            # 計算縮放比例
            scale_w = window_width / last_frame.shape[1]
            scale_h = video_area_height / last_frame.shape[0]
            scale = min(scale_w, scale_h)
            
            new_w = int(last_frame.shape[1] * scale)
            new_h = int(last_frame.shape[0] * scale)
            
            frame_surface = pygame.transform.scale(frame_surface, (new_w, new_h))
            
            # 置中顯示
            x = (window_width - new_w) // 2
            y = (video_area_height - new_h) // 2
            screen.blit(frame_surface, (x, y))
        
        # 繪製控制面板
        panel_rect = pygame.Rect(0, window_height - panel_height, window_width, panel_height)
        pygame.draw.rect(screen, COLOR_PANEL, panel_rect)
        pygame.draw.line(screen, COLOR_ACCENT, (0, window_height - panel_height), 
                        (window_width, window_height - panel_height), 2)
        
        # 狀態資訊
        status_text = f"FPS: {current_fps:.1f} | 物件: {current_objects} | 提示詞: {current_prompt}"
        if paused:
            status_text += " | [暫停]"
        status_surface = font_medium.render(status_text, True, COLOR_GREEN)
        screen.blit(status_surface, (10, window_height - 115))
        
        # 輸入框
        input_box_rect = pygame.Rect(10, window_height - 45, 300, 35)
        box_color = COLOR_ACCENT if input_active else (100, 100, 100)
        pygame.draw.rect(screen, box_color, input_box_rect, 2)
        
        display_text = input_text if input_active else "點擊輸入提示詞 (或按 Tab)"
        text_color = COLOR_TEXT if input_active else (150, 150, 150)
        text_surface = font_medium.render(display_text, True, text_color)
        screen.blit(text_surface, (15, window_height - 40))
        
        # 快速選擇按鈕
        btn_start_x = 320
        for i, prompt in enumerate(QUICK_PROMPTS[:4]):
            btn_rect = pygame.Rect(btn_start_x + i * 80, window_height - 75, 75, 25)
            btn_color = COLOR_ACCENT if prompt == current_prompt else (60, 60, 80)
            pygame.draw.rect(screen, btn_color, btn_rect, border_radius=5)
            text = font_small.render(f"{i+1}:{prompt}", True, COLOR_TEXT)
            screen.blit(text, (btn_rect.x + 5, btn_rect.y + 5))
        
        for i, prompt in enumerate(QUICK_PROMPTS[4:8]):
            btn_rect = pygame.Rect(btn_start_x + i * 80, window_height - 45, 75, 25)
            btn_color = COLOR_ACCENT if prompt == current_prompt else (60, 60, 80)
            pygame.draw.rect(screen, btn_color, btn_rect, border_radius=5)
            text = font_small.render(f"{i+5}:{prompt}", True, COLOR_TEXT)
            screen.blit(text, (btn_rect.x + 5, btn_rect.y + 5))
        
        # 說明文字
        help_text = "Q:退出 | S:截圖 | P:暫停 | 1-8:快速切換"
        help_surface = font_small.render(help_text, True, (150, 150, 150))
        screen.blit(help_surface, (window_width - 300, window_height - 25))
        
        pygame.display.flip()
        clock.tick(60)  # 限制 60 FPS 更新率
    
    # 清理
    cap.release()
    pygame.quit()
    
    print("\n" + "=" * 60)
    print(f"處理完成！共 {frame_count} 幀")
    print("=" * 60)


if __name__ == "__main__":
    main()
