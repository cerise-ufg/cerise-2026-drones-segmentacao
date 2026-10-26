# Imagem da frente DRONES (deteccao/contagem de pessoas) para o DGX Spark (GB10, aarch64)
FROM nvcr.io/nvidia/pytorch:26.05-py3
# torch/torchvision ja vem otimizados na imagem NGC: instala ultralytics SEM dependencias
# para nao substituir o torch da NVIDIA, e depois so as dependencias que faltam.
RUN pip install --no-cache-dir --no-deps ultralytics ultralytics-thop && \
    pip install --no-cache-dir opencv-python-headless polars && \
    python -c "import ultralytics, cv2; print('ultralytics', ultralytics.__version__, 'cv2', cv2.__version__)"
ENV YOLO_CONFIG_DIR=/w/ultra_cfg \
    PYTHONUNBUFFERED=1
WORKDIR /w
