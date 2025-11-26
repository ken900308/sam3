# SAM3 Docker 部署包 - 安裝完成

## ✅ 已創建的檔案

我已經為你創建了完整的 SAM3 Docker 環境，包含以下檔案：

### 📁 docker/ 目錄結構

```
sam3/docker/
├── Dockerfile                 # Docker 映像定義 (Ubuntu 22.04 + ROS Humble + CUDA 12.6)
├── docker-compose.yml         # 容器編排配置 (支援 GPU 和 X11)
├── entrypoint.sh             # 容器啟動腳本
├── run_docker.sh             # 快速啟動腳本 (推薦使用)
├── test_environment.py       # 環境測試腳本
├── .dockerignore             # Docker 構建忽略檔案
├── README.md                 # 完整使用文檔
├── QUICKSTART.md             # 快速參考指南
└── SETUP_COMPLETE.md         # 本檔案
```

---

## 🎯 主要特性

### 1. **Dockerfile** 特點
- ✅ Base Image: Ubuntu 22.04 Jammy
- ✅ ROS Humble Desktop (完整版)
- ✅ CUDA 12.6 + cuDNN
- ✅ Python 3.12
- ✅ PyTorch 2.7 with CUDA support
- ✅ 所有 SAM3 依賴預裝

### 2. **docker-compose.yml** 特點
- ✅ GPU 支援 (所有 GPU)
- ✅ X11 顯示支援 (GUI 應用)
- ✅ 本地目錄掛載 (`../` → `/workspace/sam3`)
- ✅ Hugging Face 模型快取掛載
- ✅ Host 網路模式 (ROS 通訊)
- ✅ 共享記憶體 (/dev/shm)
- ✅ 兩個服務: sam3 (主容器) + sam3-jupyter (Notebook)

### 3. **run_docker.sh** 功能
- ✅ 自動檢查依賴
- ✅ 一鍵構建和啟動
- ✅ 多種操作模式 (build/run/jupyter/shell/logs/clean)
- ✅ 彩色輸出和友好提示
- ✅ X11 自動配置

### 4. **test_environment.py** 測試
- ✅ Python 版本檢查
- ✅ PyTorch + CUDA 檢查
- ✅ ROS 環境檢查
- ✅ 依賴套件檢查
- ✅ GPU 記憶體檢查
- ✅ SAM3 組件檢查

---

## 🚀 快速開始 (3 步驟)

### 步驟 1: 準備環境

```bash
# 1. 確保 Docker 和 NVIDIA Docker 已安裝
docker --version
docker-compose --version
nvidia-smi

# 2. 允許 X11 連接
xhost +local:docker

# 3. (可選) 設置 Hugging Face Token
export HF_TOKEN="your_huggingface_token"
```

### 步驟 2: 構建映像

```bash
cd /home/kun/Desktop/projects/meta/sam3/docker

# 使用快速腳本構建 (推薦)
./run_docker.sh build

# 或使用 docker-compose
docker-compose build
```

構建時間: 約 15-30 分鐘 (取決於網路速度)

### 步驟 3: 啟動容器

```bash
# 方法 1: 使用快速腳本 (推薦)
./run_docker.sh run

# 方法 2: 使用 docker-compose
docker-compose run --rm sam3

# 進入容器後
cd /workspace/sam3
pip install -e .
python docker/test_environment.py
```

---

## 📚 使用範例

### 範例 1: 互動式使用

```bash
# 啟動互動式容器
./run_docker.sh run

# 容器內操作
cd /workspace/sam3
pip install -e .
python examples/sam3_image_predictor_example.py
```

### 範例 2: Jupyter Notebook

```bash
# 啟動 Jupyter 服務
./run_docker.sh jupyter

# 訪問 http://localhost:8888
```

### 範例 3: 執行單一命令

```bash
# 在已運行的容器中執行
./run_docker.sh exec python docker/test_environment.py

# 或啟動新容器執行
./run_docker.sh run python docker/test_environment.py
```

### 範例 4: 開發工作流

```bash
# 1. 啟動容器 (背景)
cd /home/kun/Desktop/projects/meta/sam3/docker
docker-compose up -d sam3

# 2. 進入容器
./run_docker.sh shell

# 3. 開發和測試
cd /workspace/sam3
# 編輯代碼 (在主機上編輯，容器內即時同步)
python your_script.py

# 4. 查看日誌
./run_docker.sh logs

# 5. 停止容器
./run_docker.sh stop
```

---

## 🔧 常用命令速查

```bash
# 構建映像
./run_docker.sh build

# 啟動互動式容器
./run_docker.sh run

# 啟動 Jupyter
./run_docker.sh jupyter

# 進入 shell
./run_docker.sh shell

# 執行命令
./run_docker.sh exec <command>

# 查看日誌
./run_docker.sh logs

# 測試環境
./run_docker.sh test

# 停止容器
./run_docker.sh stop

# 清理所有
./run_docker.sh clean

# 顯示幫助
./run_docker.sh help
```

---

## 📦 容器內文件結構

```
/workspace/
├── sam3/                    # SAM3 源代碼 (掛載自主機)
│   ├── sam3/               # Python 包
│   ├── examples/           # 範例腳本
│   ├── docker/             # Docker 配置
│   └── ...
├── .cache/
│   └── huggingface/        # 模型快取 (持久化)
└── datasets/               # 數據集目錄 (可選)

/opt/ros/humble/            # ROS Humble 安裝
```

---

## 🔍 驗證安裝

### 在主機上測試

```bash
cd /home/kun/Desktop/projects/meta/sam3/docker

# 測試 Docker 和 GPU
./run_docker.sh test
```

### 在容器內測試

```bash
# 啟動容器
./run_docker.sh run

# 容器內執行
python docker/test_environment.py

# 測試 SAM3 導入
python -c "from sam3.model_builder import build_sam3_image_model; print('✅ SAM3 OK')"

# 測試 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 測試 ROS
ros2 --version
```

---

## 🐛 常見問題

### Q1: GPU 不可用？

```bash
# 檢查 NVIDIA driver
nvidia-smi

# 檢查 NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# 重啟 Docker
sudo systemctl restart docker
```

### Q2: X11 顯示錯誤？

```bash
# 重新允許連接
xhost +local:docker

# 檢查 DISPLAY
echo $DISPLAY

# 測試 (在容器內)
xclock
```

### Q3: 記憶體不足？

在 `docker-compose.yml` 中添加：

```yaml
shm_size: '8gb'
```

### Q4: 權限問題？

```bash
# 以當前用戶運行
./run_docker.sh run --user "$(id -u):$(id -g)"
```

### Q5: 構建太慢？

```bash
# 使用 Docker BuildKit
export DOCKER_BUILDKIT=1
./run_docker.sh build
```

---

## 📖 更多資源

- **完整文檔**: `docker/README.md`
- **快速參考**: `docker/QUICKSTART.md`
- **SAM3 文檔**: `../SAM3_使用教學.md`
- **官方 GitHub**: https://github.com/facebookresearch/sam3
- **Hugging Face**: https://huggingface.co/facebook/sam3

---

## 🎓 進階配置

### 多 GPU 配置

編輯 `docker-compose.yml`:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 1
```

### 自定義端口

```yaml
ports:
  - "8888:8888"  # Jupyter
  - "6006:6006"  # TensorBoard
```

### 資源限制

```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 32G
```

### 持久化存儲

```yaml
volumes:
  - ./checkpoints:/workspace/checkpoints  # 模型權重
  - ./results:/workspace/results          # 結果輸出
```

---

## ✅ 檢查清單

在使用前，請確保：

- [ ] Docker 已安裝並運行
- [ ] Docker Compose 已安裝
- [ ] NVIDIA Docker Runtime 已安裝
- [ ] NVIDIA Driver 支援 CUDA 12.6+
- [ ] X11 權限已配置 (`xhost +local:docker`)
- [ ] (可選) Hugging Face Token 已設置
- [ ] 所有腳本有執行權限
- [ ] 已閱讀 `README.md` 和 `QUICKSTART.md`

---

## 🎉 恭喜！

SAM3 Docker 環境已完全配置完成！你現在可以：

1. ✅ 在隔離的容器中運行 SAM3
2. ✅ 使用 GPU 加速
3. ✅ 顯示 GUI 應用 (X11)
4. ✅ 與 ROS Humble 整合
5. ✅ 實時編輯代碼 (主機和容器同步)
6. ✅ 持久化模型和數據

---

## 📞 需要幫助？

如有問題：

1. 查看 `docker/README.md` 獲取詳細文檔
2. 運行 `./run_docker.sh help` 查看所有命令
3. 運行 `./run_docker.sh test` 診斷問題
4. 查看 GitHub Issues: https://github.com/facebookresearch/sam3/issues

---

**創建日期**: 2025-11-25  
**版本**: 1.0.0  
**維護**: Kun  
**狀態**: ✅ 生產就緒

祝你使用愉快！🚀
