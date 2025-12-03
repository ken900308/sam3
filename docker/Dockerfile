# SAM3 Docker Image
# Base: Ubuntu 24.04 Noble + ROS2 Jazzy Desktop + CUDA 12.6 + Python 3.12
# Requirements: SAM3 needs Python 3.12+, PyTorch 2.7+, CUDA 12.6+
# Note: Using Ubuntu 24.04 for native Python 3.12 and ROS2 Jazzy support

# Stage 1: Base image with CUDA 12.6 and Ubuntu 24.04
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set timezone
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    vim \
    nano \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 ships with Python 3.12 natively
# Install Python development packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (using --ignore-installed to avoid conflicts with system packages)
RUN python3 -m pip install --upgrade pip --break-system-packages --ignore-installed

# Install ROS2 Jazzy Desktop
RUN apt-get update && apt-get install -y \
    lsb-release \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Add ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Jazzy Desktop (full desktop version with native Python 3.12 support)
RUN apt-get update && apt-get install -y \
    ros-jazzy-desktop \
    ros-jazzy-ros-base \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true && \
    rosdep update

# Install additional ROS2 Jazzy packages
RUN apt-get update && apt-get install -y \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    ros-jazzy-image-transport \
    ros-jazzy-image-geometry \
    && rm -rf /var/lib/apt/lists/*

# Install X11 and GUI dependencies for display
# Note: libgl1-mesa-glx is replaced by libgl1 in Ubuntu 24.04
RUN apt-get update && apt-get install -y \
    x11-apps \
    mesa-utils \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.7 with CUDA 12.6 support
# Using --break-system-packages for Ubuntu 24.04's PEP 668 compliance
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.7.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Install SAM3 core dependencies
# Use --ignore-installed to override system-installed numpy
RUN pip install --no-cache-dir --break-system-packages --ignore-installed \
    timm>=1.0.17 \
    numpy==1.26 \
    tqdm \
    ftfy==6.1.1 \
    regex \
    iopath>=0.1.10 \
    typing_extensions \
    huggingface_hub

# Install additional dependencies for notebooks and development
# Use --ignore-installed to override system-installed packages
RUN pip install --no-cache-dir --break-system-packages --ignore-installed \
    matplotlib \
    jupyter \
    notebook \
    ipywidgets \
    ipycanvas \
    ipympl \
    pycocotools \
    decord \
    opencv-python \
    einops \
    scikit-image \
    scikit-learn \
    opencv-contrib-python \
    pillow \
    requests \
    flask

# Install development tools
RUN pip install --no-cache-dir --break-system-packages --ignore-installed \
    pytest \
    pytest-cov \
    black==24.2.0 \
    ufmt==2.8.0 \
    usort==1.0.2 \
    gitpython==3.1.31 \
    yt-dlp \
    pandas \
    numba \
    python-rapidjson \
    psutil  # Pre-install compatible psutil

# Install training dependencies
RUN pip install --no-cache-dir --break-system-packages \
    hydra-core \
    submitit \
    tensorboard \
    zstandard \
    scipy \
    torchmetrics \
    fvcore \
    fairscale

# Install Transformers for Hugging Face integration
RUN pip install --no-cache-dir --break-system-packages \
    transformers \
    accelerate

# Set up ROS environment
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

# Set up environment variables
ENV PYTHONPATH="/workspace/sam3:${PYTHONPATH}"
ENV ROS_DISTRO=jazzy
ENV ROS_VERSION=2
ENV ROS_PYTHON_VERSION=3

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Enable TensorFloat-32 for Ampere GPUs
ENV NVIDIA_TF32_OVERRIDE=1

# Create workspace directory
WORKDIR /workspace

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
