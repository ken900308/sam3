#!/bin/bash

# SAM3 Docker 快速啟動腳本
# 使用方法: ./run_docker.sh [command]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函數：打印彩色信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 檢查必要的依賴
check_dependencies() {
    print_info "檢查依賴..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安裝。請先安裝 Docker。"
        exit 1
    fi
    
    # 檢查 Docker Compose (支援 v1 和 v2)
    COMPOSE_CMD=""
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        print_info "偵測到 Docker Compose v1"
    elif docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
        print_info "偵測到 Docker Compose v2"
    else
        print_error "Docker Compose 未安裝。請先安裝 Docker Compose。"
        exit 1
    fi
    
    # 檢查 NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker 可能未正確配置。"
        print_warning "請確保已安裝 nvidia-docker2 並重啟 Docker 服務。"
    fi
    
    print_success "依賴檢查完成"
}

# 允許 X11 連接
setup_x11() {
    print_info "設置 X11 顯示權限..."
    xhost +local:docker > /dev/null 2>&1 || print_warning "無法設置 X11 權限"
}

# 顯示使用說明
show_usage() {
    cat << EOF
SAM3 Docker 快速啟動腳本

使用方法:
    ./run_docker.sh [command] [options]

命令:
    build       構建 Docker 映像
    start       啟動持久化容器（推薦，不會自動刪除）
    run         啟動一次性容器（退出後自動刪除）
    jupyter     啟動 Jupyter notebook 服務
    exec        在運行中的容器執行命令
    stop        停止所有容器
    clean       清理容器和映像                   
    logs        查看容器日誌                    not ok
    shell       進入容器 shell                 not ok
    test        測試容器配置                     ok 
    help        顯示此幫助信息                    ok

範例:
    # 構建映像
    ./run_docker.sh build

    # 啟動持久化容器（推薦）
    ./run_docker.sh start

    # 進入容器
    ./run_docker.sh shell

    # 啟動一次性容器（退出後會刪除）
    ./run_docker.sh run

    # 啟動 Jupyter notebook
    ./run_docker.sh jupyter

    # 進入運行中的容器
    ./run_docker.sh shell

    # 在容器中執行命令
    ./run_docker.sh exec python examples/test.py

    # 查看日誌
    ./run_docker.sh logs

    # 停止容器
    ./run_docker.sh stop

    # 清理所有
    ./run_docker.sh clean

環境變數:
    HF_TOKEN    - Hugging Face token (用於下載模型)
    CUDA_VISIBLE_DEVICES - 指定使用的 GPU (預設: 0)

EOF
}

# 構建映像
build_image() {
    print_info "構建 SAM3 Docker 映像..."
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD build "$@"
    print_success "映像構建完成"
}

# 運行容器（一次性，退出後刪除）
run_container() {
    setup_x11
    print_info "啟動 SAM3 容器（一次性模式，退出後會刪除）..."
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD run --rm sam3 "$@"
}

# 啟動持久化容器（不會自動刪除）
start_container() {
    setup_x11
    print_info "啟動 SAM3 持久化容器..."
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD up -d sam3
    print_success "容器已在背景啟動"
    print_info "使用 './run_docker.sh shell' 進入容器"
    print_info "使用 './run_docker.sh stop' 停止容器"
}

# 啟動 Jupyter
start_jupyter() {
    setup_x11
    print_info "啟動 Jupyter notebook..."
    print_info "訪問 http://localhost:8888 使用 notebook"
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD up sam3-jupyter
}

# 執行命令
exec_command() {
    print_info "在容器中執行命令: $*"
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD exec sam3 "$@"
}

# 進入 shell
enter_shell() {
    print_info "進入容器 shell..."
    cd "$SCRIPT_DIR"
    if $COMPOSE_CMD ps sam3 | grep -q "Up"; then
        $COMPOSE_CMD exec sam3 /bin/bash
    else
        print_warning "容器未運行，啟動新的互動式容器..."
        run_container /bin/bash
    fi
}

# 停止容器
stop_containers() {
    print_info "停止所有容器..."
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD down
    print_success "容器已停止"
}

# 清理
clean_all() {
    print_warning "這將刪除所有 SAM3 容器和映像"
    read -p "確定要繼續嗎? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "清理容器..."
        cd "$SCRIPT_DIR"
        $COMPOSE_CMD down -v
        docker rmi sam3:latest 2>/dev/null || true
        print_success "清理完成"
    else
        print_info "取消清理"
    fi
}

# 查看日誌
show_logs() {
    print_info "顯示容器日誌..."
    cd "$SCRIPT_DIR"
    $COMPOSE_CMD logs -f "$@"
}

# 測試容器
test_container() {
    print_info "測試 Docker 配置..."
    
    # 測試 NVIDIA Docker
    print_info "測試 GPU 訪問..."
    docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
    
    if [ $? -eq 0 ]; then
        print_success "GPU 訪問正常"
    else
        print_error "GPU 訪問失敗"
        return 1
    fi
    
    # 測試 X11
    print_info "測試 X11 顯示..."
    setup_x11
    
    if [ -n "$DISPLAY" ]; then
        print_success "DISPLAY 環境變數已設置: $DISPLAY"
    else
        print_warning "DISPLAY 環境變數未設置"
    fi
    
    print_success "所有測試通過"
}

# 主函數
main() {
    cd "$SCRIPT_DIR"
    
    # 檢查參數
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    COMMAND="$1"
    shift
    
    case "$COMMAND" in
        build)
            check_dependencies
            build_image "$@"
            ;;
        run)
            check_dependencies
            run_container "$@"
            ;;
        start)
            check_dependencies
            start_container "$@"
            ;;
        jupyter)
            check_dependencies
            start_jupyter "$@"
            ;;
        exec)
            exec_command "$@"
            ;;
        shell)
            enter_shell "$@"
            ;;
        stop)
            stop_containers "$@"
            ;;
        clean)
            clean_all "$@"
            ;;
        logs)
            show_logs "$@"
            ;;
        test)
            check_dependencies
            test_container "$@"
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "未知命令: $COMMAND"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# 執行主函數
main "$@"
