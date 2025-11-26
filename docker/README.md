# SAM3 Docker 使用指南

本目錄包含 SAM3 的 Docker 配置，支援 **Ubuntu 24.04**、**ROS2 Jazzy Desktop**、**CUDA 12.6** 和 **Python 3.12**。

> **注意**: 使用 Ubuntu 24.04 是因為 ROS2 Jazzy 原生支援 Python 3.12，解決了 ROS Humble 只支援 Python 3.10 的限制。

## 📋 系統需求

### 硬體要求
- NVIDIA GPU (建議 16GB+ VRAM)
- 16GB+ RAM
- 50GB+ 可用磁碟空間

### 軟體要求
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker Runtime (nvidia-docker2)
- **NVIDIA Driver 550+** (支援 CUDA 12.6+)

## 🚀 快速開始

### 1. 安裝 Docker 和 NVIDIA Docker

```bash
# 安裝 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 安裝 NVIDIA Docker Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 驗證安裝
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

### 2. 允許 X11 連接

```bash
# 允許本地 Docker 連接 X11
xhost +local:docker
```

### 3. 設置 Hugging Face Token (可選)

```bash
# 如果需要下載 SAM3 模型，設置你的 Hugging Face token
export HF_TOKEN="your_huggingface_token_here"
```

### 4. 構建 Docker 映像

```bash
cd /path/to/sam3/docker
docker-compose build
```

或使用 Docker 直接構建：

```bash
cd /path/to/sam3
docker build -f docker/Dockerfile -t sam3:latest .
```

### 5. 啟動容器

#### 選項 A: 使用 docker-compose (推薦)

```bash
cd /path/to/sam3/docker

# 啟動主容器 (互動式)
docker-compose run --rm sam3

# 或啟動 Jupyter notebook 服務
docker-compose up sam3-jupyter

# 背景運行
docker-compose up -d sam3
docker-compose exec sam3 bash
```

#### 選項 B: 使用 docker run

```bash
docker run -it --rm \
    --gpus all \
    --network host \
    --ipc=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/..:/workspace/sam3 \
    -v ~/.cache/huggingface:/workspace/.cache/huggingface \
    -v /dev/shm:/dev/shm \
    -w /workspace/sam3 \
    sam3:latest \
    /bin/bash
```

## 📦 容器內使用

### 安裝 SAM3

容器啟動後，SAM3 代碼已經掛載到 `/workspace/sam3`：

```bash
# 進入容器後
cd /workspace/sam3

# 安裝 SAM3 (editable mode)
pip install -e .

# 或安裝完整開發依賴
pip install -e ".[notebooks,dev]"
```

### 驗證安裝

```bash
# 檢查 CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"

# 檢查 ROS
printenv | grep ROS
ros2 --version

# 檢查 SAM3
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 import successful!')"
```

### 運行範例

```bash
# 運行 Python 腳本
cd /workspace/sam3
python examples/your_script.py

# 啟動 Jupyter notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 運行 ROS 節點 (如果你創建了 ROS 包)
source /opt/ros/humble/setup.bash
ros2 run your_package your_node
```

## 🎯 常用操作

### 進入正在運行的容器

```bash
# 使用 docker-compose
docker-compose exec sam3 bash

# 或使用 docker
docker exec -it sam3_container bash
```

### 查看日誌

```bash
# 查看容器日誌
docker-compose logs -f sam3

# 查看 Jupyter 日誌
docker-compose logs -f sam3-jupyter
```

### 停止和刪除容器

```bash
# 停止容器
docker-compose down

# 停止並刪除所有數據
docker-compose down -v
```

### 重新構建映像

```bash
# 重新構建（不使用快取）
docker-compose build --no-cache

# 構建特定服務
docker-compose build sam3
```

## 🔧 配置說明

### Volume 掛載

在 `docker-compose.yml` 中配置了以下掛載：

```yaml
volumes:
  # SAM3 源代碼 (可讀寫)
  - ../:/workspace/sam3
  
  # X11 顯示
  - /tmp/.X11-unix:/tmp/.X11-unix:rw
  
  # Hugging Face 模型快取
  - ~/.cache/huggingface:/workspace/.cache/huggingface
  
  # 數據集目錄 (唯讀)
  - ~/datasets:/workspace/datasets:ro
  
  # 共享記憶體
  - /dev/shm:/dev/shm
```

### 環境變數

```yaml
environment:
  - DISPLAY=${DISPLAY}                    # X11 顯示
  - NVIDIA_VISIBLE_DEVICES=all            # 所有 GPU
  - HF_TOKEN=${HF_TOKEN}                  # Hugging Face token
  - ROS_DOMAIN_ID=0                       # ROS domain ID
  - CUDA_VISIBLE_DEVICES=0                # 指定使用的 GPU
```

### 網路模式

使用 `network_mode: host` 以便：
- 容器可以訪問主機的所有網路端口
- ROS 節點可以互相發現
- 簡化網路配置

如果需要隔離網路，可以改用 bridge 模式：

```yaml
network_mode: bridge
ports:
  - "8888:8888"  # Jupyter
  - "6006:6006"  # TensorBoard
```

## 🐛 故障排除

### 問題 1: CUDA 不可用

```bash
# 檢查 NVIDIA driver
nvidia-smi

# 檢查 Docker GPU 支援
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# 重啟 Docker
sudo systemctl restart docker
```

### 問題 2: X11 顯示錯誤

```bash
# 重新允許 X11 連接
xhost +local:docker

# 檢查 DISPLAY 變數
echo $DISPLAY

# 在容器內測試
xclock  # 應該顯示一個時鐘視窗
```

### 問題 3: 記憶體不足

```bash
# 增加共享記憶體大小
docker run --shm-size=8gb ...

# 或在 docker-compose.yml 中添加
shm_size: '8gb'
```

### 問題 4: 權限問題

```bash
# 以當前用戶運行容器
docker-compose run --rm --user "$(id -u):$(id -g)" sam3

# 或修改 docker-compose.yml
user: "${UID}:${GID}"
```

### 問題 5: Hugging Face 下載失敗

```bash
# 設置 token
export HF_TOKEN="your_token"

# 或在容器內手動登錄
huggingface-cli login

# 檢查網路連接
curl https://huggingface.co
```

## 📝 進階使用

### 多 GPU 配置

```yaml
environment:
  # 使用特定 GPU
  - CUDA_VISIBLE_DEVICES=0,1

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0', '1']
          capabilities: [gpu]
```

### 資源限制

```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 32G
    reservations:
      cpus: '4'
      memory: 16G
```

### 自定義啟動腳本

修改 `entrypoint.sh` 添加自定義初始化邏輯：

```bash
#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

# 自動安裝 SAM3
if [ ! -f "/workspace/sam3/.installed" ]; then
    cd /workspace/sam3
    pip install -e .
    touch .installed
fi

# 其他初始化...

exec "$@"
```

## 📚 相關資源

- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM3 Hugging Face](https://huggingface.co/facebook/sam3)
- [ROS Humble 文檔](https://docs.ros.org/en/humble/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose](https://docs.docker.com/compose/)

## 🤝 貢獻

如有問題或改進建議，請提交 Issue 或 Pull Request。

---

**建立時間**: 2025-11-25
**維護者**: Your Name
