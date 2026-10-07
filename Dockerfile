# Inference image for Zoho Catalyst AppSail (custom runtime) or any container host: CPU only, no PyTorch.
#   docker build --platform linux/amd64 -t colony-counter:latest .
#   docker run -p 9000:9000 colony-counter:latest
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PORT=9000
WORKDIR /srv

COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

COPY app/ app/
COPY weights/colony.onnx weights/colony.json weights/

RUN useradd --uid 10001 --no-create-home colony
USER colony
EXPOSE 9000
CMD ["python", "app/server.py"]
