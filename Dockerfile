# ==============================
# Stage 1: Builder
# ==============================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# 设置 HTTP/HTTPS 代理（可选）
ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

# Isaac Lab version (v2.3.2+ required for TacSL tactile sensors)
ARG ISAACLAB_VERSION=v2.3.2

# 使用阿里云源
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list


# 安装构建依赖（GCC 12 + GLU + Vulkan）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 g++-12 cmake build-essential unzip git-lfs \
    libglu1-mesa-dev vulkan-tools  wget\
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*
# 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy
# 接受 Conda TOS + 创建环境
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n unitree_sim_env python=3.11 -y && \
    conda clean -afy

# 切换到 Conda 环境
SHELL ["conda", "run", "-n", "unitree_sim_env", "/bin/bash", "-c"]



RUN conda install -y -c conda-forge "libgcc-ng>=12" "libstdcxx-ng>=12" && \
    apt-get update && apt-get install -y libvulkan1 vulkan-tools && rm -rf /var/lib/apt/lists/*


# 安装 PyTorch（CUDA 12.6 对应）
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126


# 安装 Isaac Sim (5.1.0 required for TacSL/Isaac Lab v2.3.2+)
RUN pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 创建工作目录
RUN mkdir -p /home/code
WORKDIR /home/code

# Accept NVIDIA Omniverse EULA non-interactively
ENV OMNI_KIT_ACCEPT_EULA=yes
ENV ACCEPT_EULA=Y

# 克隆并安装 IsaacLab (v2.3.2+ includes TacSL tactile sensors)
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    git fetch --tags && \
    git checkout ${ISAACLAB_VERSION} && \
    echo "yes" | ./isaaclab.sh --install

# 安装 isaaclab_contrib (TacSL visuo-tactile sensors)
RUN cd /home/code/IsaacLab/source/isaaclab_contrib && \
    pip install -e .
    
# 构建 CycloneDDS
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x /cyclonedds && \
    cd /cyclonedds && mkdir build install && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target install

# 设置 CycloneDDS 环境变量
ENV CYCLONEDDS_HOME=/cyclonedds/install

# 安装 unitree_sdk2_python
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python && \
    cd unitree_sdk2_python && pip install -e .

# 克隆 unitree_sim_isaaclab
RUN git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git /home/code/unitree_sim_isaaclab && \
    cd /home/code/unitree_sim_isaaclab && pip install -r requirements.txt


# ==============================
# Stage 2: Runtime
# ==============================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# 禁止 Isaac Sim 自动启动
ENV OMNI_KIT_ALLOW_ROOT=1
ENV OMNI_KIT_DISABLE_STARTUP=1

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa git-lfs zenity unzip libxt6 \
    && rm -rf /var/lib/apt/lists/*

# 复制 Conda 环境和代码
COPY --from=builder /home/code/IsaacLab /home/code/IsaacLab
COPY --from=builder /home/code/unitree_sdk2_python /home/code/unitree_sdk2_python

COPY --from=builder /cyclonedds /cyclonedds
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /home/code/unitree_sim_isaaclab /home/code/unitree_sim_isaaclab


ENV CYCLONEDDS_HOME=/cyclonedds/install

# Re-install editable packages (symlinks break when copying between stages)
SHELL ["conda", "run", "-n", "unitree_sim_env", "/bin/bash", "-c"]

# Install setuptools (provides pkg_resources needed by flatdict 4.0.1)
RUN pip install --upgrade "setuptools>=65.0.0" wheel

# Install IsaacLab packages (flatdict 4.0.1 will build correctly now)
RUN cd /home/code/IsaacLab/source/isaaclab && pip install -e . && \
    cd /home/code/IsaacLab/source/isaaclab_tasks && pip install -e . && \
    cd /home/code/IsaacLab/source/isaaclab_contrib && pip install -e . && \
    cd /home/code/unitree_sdk2_python && pip install -e .

# Install teleimager submodule
RUN cd /home/code/unitree_sim_isaaclab && \
    git config --global --add safe.directory /home/code/unitree_sim_isaaclab && \
    git submodule update --init --recursive && \
    cd teleimager && \
    sed -i 's|requires-python = ">=3.8,<3.11"|requires-python = ">=3.8,<3.12"|' pyproject.toml && \
    pip install -e . && \
    pip install aiortc aiohttp

# 写入 bashrc 初始化
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate unitree_sim_env' >> ~/.bashrc && \
    echo 'export OMNI_KIT_ALLOW_ROOT=1' >> ~/.bashrc && \
    echo 'export OMNI_KIT_DISABLE_STARTUP=1' >> ~/.bashrc

WORKDIR /home/code

# Generate TacSL elastomer USD (optional - can also be done at runtime)
# Uncomment to pre-generate tactile sensor meshes:
# RUN conda run -n unitree_sim_env python /home/code/unitree_sim_isaaclab/tools/create_tactile_elastomers.py \
#     --input /home/code/unitree_sim_isaaclab/assets/robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd \
#     --output /home/code/unitree_sim_isaaclab/assets/robots/g1-29dof-inspire-tactile/g1_29dof_with_inspire_tactile.usd

# 默认进入 Conda 环境 bash
CMD ["conda", "run", "-n", "unitree_sim_env", "/bin/bash"]

# ==============================
# TacSL Tactile Sensor Usage
# ==============================
# After container startup, generate tactile USD (if not pre-generated):
#   cd /home/code/unitree_sim_isaaclab
#   python tools/create_tactile_elastomers.py \
#       --input assets/robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd \
#       --output assets/robots/g1-29dof-inspire-tactile/g1_29dof_with_inspire_tactile.usd
#
# Run simulation with TacSL tactile sensors:
#   python sim_main.py --task Isaac-PickPlace-Cylinder-G129-Inspire-TacSL --enable_inspire_dds
#
# Verify TacSL is available:
#   python -c "from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg; print('TacSL OK')"