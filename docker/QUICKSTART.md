# SAM3 Docker 快速參考

## 🚀 快速啟動命令

```bash
cd /home/kun/Desktop/projects/meta/sam3/docker

# 1. 構建映像 (首次使用)
./run_docker.sh build

# 2. 啟動容器
./run_docker.sh run

# 3. 啟動 Jupyter
./run_docker.sh jupyter

# 4. 進入容器
./run_docker.sh shell
```

## 📦 容器內操作

```bash
# 安裝 SAM3
cd /workspace/sam3
pip install -e .

# 測試環境
python docker/test_environment.py

# 運行範例
python examples/sam3_image_predictor_example.py

# 啟動 Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 🔧 常用命令

```bash
# 查看日誌
./run_docker.sh logs

# 停止容器
./run_docker.sh stop

# 測試配置
./run_docker.sh test

# 清理所有
./run_docker.sh clean
```

## 📋 檢查清單

- [ ] 安裝 Docker 和 Docker Compose
- [ ] 安裝 NVIDIA Docker Runtime
- [ ] 設置 X11 權限: `xhost +local:docker`
- [ ] 設置 Hugging Face Token (可選): `export HF_TOKEN="your_token"`
- [ ] 構建映像: `./run_docker.sh build`
- [ ] 測試環境: `./run_docker.sh test`
- [ ] 啟動容器: `./run_docker.sh run`

## 🐛 故障排除

### GPU 不可用
```bash
# 檢查 NVIDIA driver
nvidia-smi

# 重啟 Docker
sudo systemctl restart docker

# 測試 GPU
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### X11 顯示問題
```bash
# 重新允許連接
xhost +local:docker

# 檢查 DISPLAY
echo $DISPLAY
```

### 記憶體不足
```bash
# 增加共享記憶體
# 編輯 docker-compose.yml 添加:
shm_size: '8gb'
```

## 📚 檔案結構

```
sam3/docker/
├── Dockerfile              # Docker 映像定義
├── docker-compose.yml      # 容器編排配置
├── entrypoint.sh          # 容器啟動腳本
├── run_docker.sh          # 快速啟動腳本
├── test_environment.py    # 環境測試腳本
├── .dockerignore          # Docker 忽略檔案
└── README.md              # 完整文檔
```

## 🌐 網路端口

- `8888`: Jupyter Notebook
- `6006`: TensorBoard (如需使用)
- ROS: 使用 host 網路模式

## 💾 Volume 掛載

- `/workspace/sam3`: SAM3 源代碼 (可讀寫)
- `/workspace/.cache/huggingface`: 模型快取
- `/workspace/datasets`: 數據集目錄 (唯讀)
- `/tmp/.X11-unix`: X11 顯示

## 🔐 環境變數

```bash
# Hugging Face Token
export HF_TOKEN="your_token"

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0

# ROS Domain
export ROS_DOMAIN_ID=0
```

## 📞 獲取幫助

```bash
# 顯示完整幫助
./run_docker.sh help

# 查看 README
cat README.md

# 測試環境
python docker/test_environment.py
```

---

**最後更新**: 2025-11-25
