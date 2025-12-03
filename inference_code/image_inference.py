#!/usr/bin/env python3
"""
SAM3 圖片推論腳本
================
基於官方 examples/sam3_image_predictor_example.ipynb 修改

使用方法:
    python image_inference.py --image <圖片路徑> --prompt "要分割的物件"

範例:
    python image_inference.py --image ../assets/images/test_image.jpg --prompt "shoe"
    python image_inference.py --image ../assets/images/truck.jpg --prompt "wheel"
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 後端
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 確保可以 import sam3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

# 設定輸出目錄
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 設定 TF32 以提升 Ampere GPU 效能
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 圖片推論")
    parser.add_argument(
        "--image", 
        type=str, 
        default=None,
        help="輸入圖片路徑"
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
        "--output", 
        type=str, 
        default=None,
        help="輸出圖片路徑 (預設自動生成)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="運算裝置 (cuda/cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SAM3 圖片推論")
    print("=" * 60)
    
    # 檢查 GPU
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  使用 CPU (速度較慢)")
    
    # 設定圖片路徑
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    if args.image is None:
        # 使用預設測試圖片
        image_path = os.path.join(sam3_root, "assets", "images", "test_image.jpg")
    else:
        image_path = args.image
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)
    
    if not os.path.exists(image_path):
        print(f"❌ 找不到圖片: {image_path}")
        sys.exit(1)
    
    print(f"\n📷 載入圖片: {image_path}")
    image = Image.open(image_path)
    print(f"   尺寸: {image.size}")
    
    # 載入模型
    print("\n🔧 載入模型...")
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_sam3_image_model(bpe_path=bpe_path, device=args.device)
        processor = Sam3Processor(model, confidence_threshold=args.confidence)
        
        # 設定圖片
        print("🖼️  處理圖片...")
        inference_state = processor.set_image(image)
        
        # 使用文字提示進行分割
        print(f"🔍 使用文字提示: '{args.prompt}'")
        inference_state = processor.set_text_prompt(
            state=inference_state, 
            prompt=args.prompt
        )
    
    # 取得結果
    masks = inference_state.get("masks")
    boxes = inference_state.get("boxes")
    scores = inference_state.get("scores")
    
    num_objects = len(masks) if masks is not None else 0
    print(f"\n📊 分割結果:")
    print(f"   找到 {num_objects} 個物件")
    if scores is not None and len(scores) > 0:
        scores_list = scores.cpu().tolist() if hasattr(scores, 'cpu') else scores
        print(f"   分數: {[f'{s:.3f}' for s in scores_list]}")
    
    # 儲存視覺化結果
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        prompt_safe = args.prompt.replace(" ", "_")[:20]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{prompt_safe}_result.png")
    else:
        output_path = args.output
    
    plot_results(image, inference_state)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 結果已儲存到: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
