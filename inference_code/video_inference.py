#!/usr/bin/env python3
"""
SAM3 視訊推論腳本
================
基於官方 examples/sam3_video_predictor_example.ipynb 修改

使用方法:
    python video_inference.py --video <視訊路徑> --prompt "要追蹤的物件"

範例:
    python video_inference.py --video ../assets/videos/0001 --prompt "person"
    python video_inference.py --video my_video.mp4 --prompt "car"

輸入格式:
    - MP4 視訊檔案
    - 或 JPEG 影格資料夾 (檔名格式: 00000.jpg, 00001.jpg, ...)
"""

import argparse
import glob
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 確保可以 import sam3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sam3
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

# 設定輸出目錄
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 設定 TF32 以提升 Ampere GPU 效能
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 視訊推論")
    parser.add_argument(
        "--video", 
        type=str, 
        default=None,
        help="輸入視訊路徑 (MP4 檔案或 JPEG 資料夾)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="person",
        help="文字提示 (要追蹤的物件)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="輸出視訊路徑 (預設自動生成)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="是否儲存每一幀為圖片"
    )
    parser.add_argument(
        "--vis-stride",
        type=int,
        default=30,
        help="視覺化間隔 (每 N 幀儲存一張)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="使用的 GPU (例如: '0' 或 '0,1')"
    )
    return parser.parse_args()


def load_video_frames(video_path):
    """載入視訊幀 (支援 MP4 和 JPEG 資料夾)"""
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    else:
        frames = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            frames.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except ValueError:
            frames.sort()
        return frames


def propagate_in_video(predictor, session_id):
    """在視訊中傳播分割結果"""
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def save_video_with_masks(video_frames, outputs_per_frame, output_path, fps=24):
    """將分割結果儲存為視訊"""
    # 準備遮罩 (prepare_masks_for_visualization 會修改原始 dict)
    # 格式: {frame_idx: {obj_id: binary_mask, ...}}
    formatted_outputs = prepare_masks_for_visualization(outputs_per_frame.copy())
    
    # 取得視訊尺寸
    if isinstance(video_frames[0], str):
        sample = cv2.imread(video_frames[0])
        height, width = sample.shape[:2]
    else:
        height, width = video_frames[0].shape[:2]
    
    # 建立視訊寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 寫入每一幀
    for frame_idx in range(len(video_frames)):
        # 載入原始幀
        if isinstance(video_frames[frame_idx], str):
            frame = cv2.imread(video_frames[frame_idx])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = video_frames[frame_idx]
        
        # 疊加遮罩
        if frame_idx in formatted_outputs:
            masks_dict = formatted_outputs[frame_idx]
            # masks_dict 格式: {obj_id: binary_mask (np.ndarray)}
            for obj_id, mask in masks_dict.items():
                if mask is not None and mask.any():
                    # 生成顏色 (根據 obj_id)
                    color = plt.cm.rainbow(obj_id / 10)[:3]
                    color = np.array(color) * 255
                    
                    # 疊加遮罩
                    mask_bool = mask.astype(bool)
                    overlay = frame.copy()
                    overlay[mask_bool] = overlay[mask_bool] * 0.5 + color * 0.5
                    frame = overlay.astype(np.uint8)
        
        # 寫入幀
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
    print(f"✅ 視訊已儲存到: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SAM3 視訊推論")
    print("=" * 60)
    
    # 設定 GPU
    if args.gpus is not None:
        gpus_to_use = [int(g) for g in args.gpus.split(",")]
    else:
        gpus_to_use = list(range(torch.cuda.device_count()))
    
    if len(gpus_to_use) > 0 and torch.cuda.is_available():
        print(f"✅ 使用 GPU: {gpus_to_use}")
    else:
        print("⚠️  無可用 GPU")
        sys.exit(1)
    
    # 設定視訊路徑
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    if args.video is None:
        video_path = os.path.join(sam3_root, "assets", "videos", "0001")
    else:
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.abspath(video_path)
    
    if not os.path.exists(video_path):
        print(f"❌ 找不到視訊: {video_path}")
        sys.exit(1)
    
    print(f"\n📹 載入視訊: {video_path}")
    video_frames = load_video_frames(video_path)
    print(f"   共 {len(video_frames)} 幀")
    
    # 建立 predictor
    print("\n🔧 載入模型...")
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    
    # 開始 session
    print("🎬 開始推論 session...")
    start_time = time.time()
    
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    
    # 添加文字提示
    print(f"🔍 使用文字提示: '{args.prompt}'")
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=args.prompt,
        )
    )
    
    initial_output = response["outputs"]
    num_objects = len(initial_output.get("obj_ids", []))
    print(f"   在第 0 幀找到 {num_objects} 個物件")
    
    # 傳播到整個視訊
    print("📤 傳播到整個視訊...")
    outputs_per_frame = propagate_in_video(predictor, session_id)
    
    elapsed = time.time() - start_time
    fps = len(video_frames) / elapsed
    print(f"\n⏱️  總耗時: {elapsed:.2f} 秒 ({fps:.2f} FPS)")
    
    # 儲存結果
    if args.output is None:
        base_name = os.path.basename(video_path.rstrip("/"))
        prompt_safe = args.prompt.replace(" ", "_")[:20]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{prompt_safe}_result.mp4")
    else:
        output_path = args.output
    
    print(f"\n💾 儲存結果...")
    save_video_with_masks(video_frames, outputs_per_frame, output_path)
    
    # 儲存關鍵幀
    if args.save_frames:
        frames_dir = os.path.join(OUTPUT_DIR, f"{base_name}_{prompt_safe}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        formatted_outputs = prepare_masks_for_visualization(outputs_per_frame)
        for frame_idx in range(0, len(video_frames), args.vis_stride):
            plt.figure(figsize=(10, 8))
            visualize_formatted_frame_output(
                frame_idx,
                video_frames,
                outputs_list=[formatted_outputs],
                titles=[f"Frame {frame_idx}"],
                figsize=(10, 8),
            )
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"✅ 關鍵幀已儲存到: {frames_dir}")
    
    # 關閉 session
    predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    predictor.shutdown()
    
    print("\n" + "=" * 60)
    print("推論完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
