# =============================================================================
# URCA_PROJECTS — VLM Safety Monitor + Latency Benchmark
# CUDA 13.0 · Python 3.13 · PyTorch 2.14 · PySide6 (headless via Xvfb)
# =============================================================================
# Base: Ubuntu 24.04 ships glibc 2.39 which is required by Python 3.13 wheels.
# CUDA 13.0.3 is confirmed on Docker Hub with full sm_120 (RTX 5090 / Blackwell) support.
# Tag format: X.Y.Z-cudnn-devel-ubuntuVV.VV  (no '9' suffix for 13.x images).
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

# --------------------------------------------------------------------------- #
# 0. Hardware compatibility fix for GeForce RTX GPUs (sm_120 / RTX 5090)
# --------------------------------------------------------------------------- #
# cuda-compat is designed strictly for Datacenter GPUs (A100/H100).
# On GeForce cards, it attempts forward compatibility and crashes with CUDA Error 804.
# Removing compat libraries forces PyTorch to use the host's native NVIDIA driver.
ENV CUDNN_FORWARD_COMPAT_DISABLE=1
RUN rm -rf /usr/local/cuda/compat /usr/local/cuda-*/compat /etc/ld.so.conf.d/*cuda-compat*.conf 2>/dev/null && ldconfig || true

# --------------------------------------------------------------------------- #
# 1. OS-level packages + Python 3.13
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.13 from deadsnakes PPA
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-dev python3.13-venv \
    # curl for health checks / downloads
    curl \
    # Build tools
    git wget build-essential cmake pkg-config \
    # OpenCV / video codec dependencies
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libgl1 libglx-mesa0 libegl1 \
    # PySide6 / Qt runtime dependencies
    libxcb1 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-shape0 libxcb-xkb1 libxkbcommon-x11-0 \
    libdbus-1-3 libfontconfig1 libfreetype6 \
    # Virtual display for headless Qt (xdpyinfo is part of x11-utils — used by entrypoint)
    xvfb x11-utils \
    # Misc
    ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create a Python 3.13 virtual environment — the correct way to use pip on Ubuntu 24.04.
# Ubuntu enforces PEP 668 (externally-managed-environment); a venv sidesteps all conflicts.
RUN python3.13 -m venv /opt/venv

# Prepend the venv to PATH so every subsequent RUN, CMD and ENTRYPOINT uses it automatically.
ENV PATH="/opt/venv/bin:$PATH"

# --------------------------------------------------------------------------- #
# 2. Working directory layout
# --------------------------------------------------------------------------- #
WORKDIR /workspace

# Copy the full project (see .dockerignore for exclusions)
COPY . /workspace/

# --------------------------------------------------------------------------- #
# 3. Python dependencies
# --------------------------------------------------------------------------- #
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch with CUDA 13.0 — must come first to anchor the CUDA build variant
# TORCH_CUDA_ARCH_LIST — must be set at install time so sm_120 (RTX 5090 Blackwell) kernels
# are selected when PyTorch JIT-compiles any custom extensions at runtime.
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"

RUN pip install --no-cache-dir \
    torch==2.14.1+cu130 \
    torchvision==0.29.1+cu130 \
    torchaudio==2.11.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

# All other dependencies — exact versions matching the Windows environment
# (these all have Python 3.13 cp313 wheels on PyPI)
RUN pip install --no-cache-dir \
    numpy==2.5.2 \
    pandas==3.0.6 \
    scipy==1.18.1 \
    scikit-learn==1.9.1 \
    matplotlib==3.11.2 \
    seaborn==0.13.2 \
    sympy==1.14.0 \
    mpmath==1.3.0 \
    networkx==3.6.1 \
    pillow==12.3.0 \
    opencv-python-headless==5.0.0.93 \
    transformers==5.17.0 \
    accelerate==1.15.0 \
    bitsandbytes==0.50.2 \
    qwen-vl-utils==0.0.14 \
    av==18.1.0 \
    tokenizers==0.23.2 \
    safetensors==0.8.0 \
    huggingface-hub==1.33.0 \
    tqdm==4.70.1 \
    psutil==7.2.2 \
    pyyaml==6.0.3 \
    regex==2026.9.10 \
    filelock==3.32.3 \
    fsspec==2026.7.0 \
    packaging==26.3 \
    typing_extensions==4.16.0 \
    click==8.5.0 \
    httpx==0.28.1 \
    httpcore==1.0.9 \
    requests==2.34.2 \
    urllib3==2.7.0 \
    certifi==2026.4.22 \
    charset-normalizer==3.4.7 \
    idna==3.15 \
    PySide6==6.11.1 \
    pytest==8.4.2

# --------------------------------------------------------------------------- #
# 4. Environment variables
# --------------------------------------------------------------------------- #
# HuggingFace cache — model weights downloaded here at first run, volume-mounted
# HuggingFace cache — model weights downloaded here at first run, volume-mounted
ENV HF_HOME=/workspace/.hf_cache
ENV TRANSFORMERS_CACHE=/workspace/.hf_cache
ENV HF_DATASETS_CACHE=/workspace/.hf_cache/datasets

# Default model snapshots (mounted from host HF cache)
ENV MODEL_PATH=/workspace/.hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
ENV JEV_MODEL_PATH=/workspace/.hf_cache/hub/models--OmniJev--OneJev-0.8B/snapshots/c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8

# Qt / PySide6 — headless via Xvfb started in entrypoint
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:99

# CUDA — expose all GPUs
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Python path — both HERMES and InfiniPot-V engines are importable
ENV PYTHONPATH=/workspace/vlm_test_ui:/workspace/temper/vlm_test_ui:/workspace

# --------------------------------------------------------------------------- #
# 5. Entrypoint
# --------------------------------------------------------------------------- #
COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
