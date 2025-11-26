# SAM3 Docker 環境設置筆記

## 📋 環境概述

| 項目 | 版本 |
|------|------|
| Base Image | `nvidia/cuda:12.6.0-devel-ubuntu24.04` |
| Ubuntu | 24.04 (Noble) |
| ROS2 | Jazzy |
| Python | 3.12.3 (系統內建) |
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.6 (container) / 13.0 (host driver) |
| GPU | NVIDIA GeForce RTX 4060 Ti 16GB |

## ✅ 已完成的設置

### 1. Docker 環境建置
- [x] Ubuntu 24.04 + ROS2 Jazzy 基礎映像
- [x] CUDA 12.6 + cuDNN 支援
- [x] PyTorch 2.7.0 with CUDA 12.6 wheel
- [x] X11 顯示支援 (GPU 加速渲染)
- [x] GPU passthrough 正常運作

### 2. ROS2 Jazzy 整合
- [x] ROS2 Jazzy Desktop 完整安裝
- [x] `rclpy` Python 綁定正常運作
- [x] `ros2` CLI 工具可用
- [x] cv_bridge, vision_msgs, image_transport 等視覺相關套件

### 3. Python 套件
- [x] SAM3 核心依賴 (timm, numpy, tqdm, etc.)
- [x] 開發工具 (pytest, black, ufmt)
- [x] Jupyter Notebook 支援
- [x] transformers, accelerate

## ⚠️ 已解決的問題

### 問題 1: ROS2 Humble 與 Python 3.12 不兼容
**原因**: ROS2 Humble 是為 Python 3.10 編譯的，SAM3 需要 Python 3.12+

**解決方案**: 改用 Ubuntu 24.04 + ROS2 Jazzy（原生支援 Python 3.12）

### 問題 2: Ubuntu 24.04 套件名稱變更
**舊套件** → **新套件**
- `libgl1-mesa-glx` → `libgl1`
- `libglib2.0-0` → `libglib2.0-0t64`

### 問題 3: 系統 Python 套件無法覆蓋
**原因**: Ubuntu 24.04 的 apt 安裝的 Python 套件沒有 RECORD 文件

**解決方案**: 在 pip install 時使用 `--ignore-installed` 和 `--break-system-packages`

### 問題 4: Docker Compose v2 語法
**原因**: 系統使用 Docker Compose v2 (plugin)，不是獨立的 docker-compose

**解決方案**: 使用 `docker compose` 而非 `docker-compose`

### 問題 5: NVIDIA Driver 版本不足
**原因**: 原驅動 550 只支援 CUDA 12.4，SAM3 需要 CUDA 12.6+

**解決方案**: 升級到 NVIDIA Driver 580.95.05（支援 CUDA 13.0）

## 🚀 快速開始

### 啟動容器
```bash
cd /home/kun/Desktop/projects/meta/sam3/docker
./run_docker.sh start    # 啟動持久化容器
./run_docker.sh shell    # 進入容器 shell
```

### 在容器內安裝 SAM3
```bash
cd /workspace/sam3
pip install -e . --break-system-packages
```

### 測試環境
```bash
# 測試 Python 和 PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 測試 ROS2
source /opt/ros/jazzy/setup.bash
ros2 --help

# 測試 SAM3
python3 -c "import sam3; print('SAM3 OK')"
```

## 📁 目錄結構

```
/workspace/
├── sam3/              # SAM3 源碼 (從 host mount)
│   ├── sam3/          # Python package
│   ├── examples/      # Jupyter notebooks
│   └── pyproject.toml
└── datasets/          # 數據集目錄
```

## 🔧 Container 管理

| 命令 | 說明 |
|------|------|
| `./run_docker.sh build` | 重新建置映像 |
| `./run_docker.sh start` | 啟動持久化容器 |
| `./run_docker.sh shell` | 進入容器 shell |
| `./run_docker.sh stop` | 停止容器 |
| `./run_docker.sh run` | 一次性執行（退出後刪除） |

## 📝 待解決 / 注意事項

### 依賴版本警告（非阻塞）
pip 報告一些版本衝突，但不影響運行：
- `torchaudio 2.7.0+cu126 requires torch==2.7.0` (實際安裝了 2.9.1)
- `colcon-core 0.20.1 requires setuptools<80` (實際安裝了 80.9.0)
- `opencv-python 4.12.0.88 requires numpy<2.3.0` (實際安裝了 2.3.5)

這些是警告，不影響 SAM3 核心功能。

### 在容器內使用 VS Code
1. 使用 Remote - Containers 擴展
2. 或在容器內啟動 code-server

## 📅 更新記錄

- **2025-11-25**: 完成 Ubuntu 24.04 + ROS2 Jazzy 遷移
- **2025-11-25**: 升級 NVIDIA Driver 550 → 580
- **2025-11-25**: 修復所有 Ubuntu 24.04 套件兼容性問題
