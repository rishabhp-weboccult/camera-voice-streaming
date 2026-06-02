# CHANGE THIS TO linux/amd64 IF YOU ARE USING THE x86_64 WHEEL!
# FROM --platform=linux/arm64 python:3.10-slim
FROM nvcr.io/nvidia/l4t-jetpack:r35.2.1

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1. Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    bzip2 \
    ca-certificates \
    cmake \
    ffmpeg \
    g++ \
    gcc \
    git \
    htop \
    libasound2 \
    libasound2-plugins \
    libgl1 \
    libglib2.0-0 \
    libsdl2-2.0-0 \
    libsm6 \
    libsndfile1 \
    libxext6 \
    libxrender1 \
    nano \
    portaudio19-dev \
    pulseaudio \
    pulseaudio-utils \
    supervisor \
    wget \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# 2. Install Miniconda (ARM64)
# --------------------------
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda init bash

# --------------------------
# 3. Create Conda Env
# --------------------------
RUN conda create -n docker_gpu_env python=3.8 -y
COPY requirements.txt /workspace/requirements.txt
COPY wheels /workspace/wheels
# --------------------------
# 4. Install Python deps
# --------------------------
SHELL ["/bin/bash", "-c"]
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate docker_gpu_env && \
    pip install --upgrade pip && \
    # pip install /workspace/wheels/cloudbox-1.0.1-py3-none-any.whl && \
    pip install /workspace/wheels/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl && \
    pip install /workspace/wheels/onnxsim-0.4.36-cp38-cp38-linux_aarch64.whl && \
    pip install onnx==1.16.0 && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# 6. Copy project files
COPY . /workspace

# CMD ["python", "inference.py"]

# 7. Install Filebrowser (ARM64)
# (Note: If you change the base image to amd64, you must also change this download link to the amd64 version of filebrowser)
RUN wget -O /tmp/filebrowser.tar.gz https://github.com/filebrowser/filebrowser/releases/download/v2.51.2/linux-arm64-filebrowser.tar.gz && \
    tar -zxvf /tmp/filebrowser.tar.gz -C /usr/local/bin filebrowser && \
    chmod +x /usr/local/bin/filebrowser && \
    rm /tmp/filebrowser.tar.gz

# 8. Filebrowser preconfiguration
RUN mkdir -p /app/logs && \
    filebrowser config init --database /app/filebrowser.db && \
    filebrowser config set --address 0.0.0.0 --port 85 --root /app --database /app/filebrowser.db --auth.method=noauth && \
    filebrowser users add admin adminpassword123 --perm.admin

# 9. Supervisor
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 85
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]