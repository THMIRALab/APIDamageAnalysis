FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System tools
RUN apt-get update && apt-get install -y \
    curl git wget build-essential \
    libglib2.0-0 libegl1-mesa-dev llvm-dev ninja-build\
    ffmpeg \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"


RUN conda create -n dasch-enigma-latest python=3.11 -y && \
    echo "source activate dasch-enigma-latest" > ~/.bashrc && \
    conda clean -afy
ENV PATH="/opt/conda/envs/dasch-enigma-latest/bin:$PATH"


#RUN conda run -n dasch-enigma-latest pip install --no-cache-dir torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio --index-url https://download.pytorch.org/whl/cu126
#RUN conda run -n dasch-enigma-latest pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN conda run -n dasch-enigma-latest pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN conda run -n dasch-enigma-latest pip install --no-cache-dir -r requirements.txt

# Copy your app
WORKDIR /workspace
COPY . .

EXPOSE 8000
CMD ["conda", "run", "--no-capture-output", "-n", "dasch-enigma-latest", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]