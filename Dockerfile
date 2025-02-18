FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Set environment variables to avoid interactive installation prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install necessary dependencies
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y wget bzip2 ca-certificates sudo git && \
    rm -rf /var/lib/apt/lists/*

# Create a new user to avoid running applications as the root user
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

# Switch to the new user
USER docker
WORKDIR /home/docker

# Download and install Miniconda
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/docker/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set conda environment variables
ENV PATH=/home/docker/miniconda/bin:$PATH

# Initialize conda and configure the Tsinghua University mirror source(If you are in China, you can use this mirror source)
RUN conda init bash && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --set show_channel_urls yes

# Create a new conda environment and install PyTorch and CUDA 12.0 toolkit
RUN conda create -n pytorch_env python=3.9 -y && \
    conda install -n pytorch_env pytorch torchvision torchaudio cuda-toolkit=12.0 -c pytorch -c nvidia -y

# Use conda run to execute commands in the specified environment
RUN conda run -n pytorch_env python -c "import torch; print(torch.__version__)"

# Set the default shell to bash
CMD ["/bin/bash"]

# # clone PatchPRO
# RUN git clone https://github.com/CRhapsody/PatchPRO.git

# # Set the working directory
# WORKDIR /home/docker/PatchPRO