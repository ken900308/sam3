#!/bin/bash
set -e

# Source ROS2 Jazzy setup
source /opt/ros/jazzy/setup.bash

# Source local workspace setup if it exists
if [ -f "/workspace/sam3/install/setup.bash" ]; then
    source /workspace/sam3/install/setup.bash
fi

# Set up X11 permissions
if [ -n "$DISPLAY" ]; then
    xhost +local:docker > /dev/null 2>&1 || true
fi

# ============================================================
# Auto-install SAM3 on first container start
# ============================================================
SAM3_INSTALLED_FLAG="/root/.sam3_installed"

if [ ! -f "$SAM3_INSTALLED_FLAG" ]; then
    echo "=================================================="
    echo "First-time setup: Installing SAM3..."
    echo "=================================================="
    
    # Install SAM3 in development mode
    if [ -f "/workspace/sam3/pyproject.toml" ]; then
        echo "Installing SAM3 with pip install -e ..."
        cd /workspace/sam3
        pip install -e . --break-system-packages 2>/dev/null || pip install -e .
        echo "SAM3 installation completed!"
    else
        echo "Warning: SAM3 source not found at /workspace/sam3"
        echo "Make sure the sam3 directory is mounted correctly."
    fi
    
    # Create flag file to mark installation complete
    touch "$SAM3_INSTALLED_FLAG"
    echo ""
fi

# ============================================================
# Print environment info
# ============================================================
echo "=================================================="
echo "SAM3 Docker Container (Ubuntu 24.04 + ROS2 Jazzy)"
echo "=================================================="
echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
echo "Python Version: $(python3 --version)"
echo "PyTorch Version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
if python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "GPU Count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
    echo "GPU Name: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo "ROS Distro: $ROS_DISTRO"
echo "Working Directory: $(pwd)"
echo "=================================================="

# Test SAM3 import
if python3 -c "import sam3" 2>/dev/null; then
    echo "✓ SAM3 module is available"
else
    echo "⚠ SAM3 module not found. Run: pip install -e /workspace/sam3"
fi

# Test ROS2 availability
if command -v ros2 &> /dev/null; then
    echo "✓ ROS2 CLI is available"
else
    echo "⚠ ROS2 CLI not found"
fi

echo "=================================================="
echo ""

# Execute the command passed to docker run
exec "$@"
