# syntax=docker/dockerfile:1
# ==============================================================================
# Imagem da API de detecção de objetos (YOLOv8 + FastAPI)
#   base    -> Python + dependências de requirements.txt (compartilhado)
#   test    -> base + requirements-dev.txt + suíte pytest
#   runtime -> imagem de produção (último estágio: é o que o Render constrói)
# ==============================================================================

# ------------------------------------------------------------------------------
# ESTÁGIO 1: base
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libglib2.0-0 é usada pelo OpenCV; o restante já vem embutido nas wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Redes corporativas com inspeção de HTTPS reassinam os certificados com uma CA
# própria, que o container não conhece. O script "com-ca-corporativa" roda um
# comando confiando também nessa CA, desde que ela seja entregue ao build com
#   docker build --secret id=corp_ca,src=corp-ca.crt ...
# Sem o secret (caso do Render) o comando roda normalmente. O bundle temporário
# é apagado ao final, então a CA não fica gravada em nenhuma camada da imagem.
# O script é gerado com printf para ter quebras de linha LF mesmo num checkout
# do Windows.
RUN printf '%s\n' \
        '#!/bin/sh' \
        'set -e' \
        'if [ -f /run/secrets/corp_ca ]; then' \
        '  cat /etc/ssl/certs/ca-certificates.crt /run/secrets/corp_ca > /tmp/ca.pem' \
        '  export PIP_CERT=/tmp/ca.pem SSL_CERT_FILE=/tmp/ca.pem REQUESTS_CA_BUNDLE=/tmp/ca.pem' \
        'fi' \
        'status=0' \
        '"$@" || status=$?' \
        'rm -f /tmp/ca.pem' \
        'exit $status' \
        > /usr/local/bin/com-ca-corporativa \
    && chmod +x /usr/local/bin/com-ca-corporativa

# Copiar só o requirements primeiro aproveita o cache do Docker: as libs pesadas
# (torch, ultralytics) só são reinstaladas quando o requirements.txt mudar.
COPY requirements.txt .
RUN --mount=type=secret,id=corp_ca \
    com-ca-corporativa pip install -r requirements.txt

# O ultralytics depende de opencv-python (com GUI), que exige libGL e disputa o
# módulo cv2 com o headless. Removemos os dois e reinstalamos apenas o headless,
# na mesma versão fixada no requirements.txt.
RUN --mount=type=secret,id=corp_ca \
    pip uninstall -y opencv-python opencv-python-headless \
    && com-ca-corporativa pip install --no-deps "$(grep -E '^opencv-python-headless==' requirements.txt | tr -d '\r')" \
    && python -c "import cv2, torch; print('OpenCV', cv2.__version__, '| Torch', torch.__version__)"

# ------------------------------------------------------------------------------
# ESTÁGIO 2: test  ->  docker build --target test -t detector-api-test .
# ------------------------------------------------------------------------------
FROM base AS test

# requirements-dev.txt começa com "-r requirements.txt"; reinstalá-lo traria o
# opencv-python de volta. Instalamos só as linhas de teste (pytest, httpx).
COPY requirements-dev.txt .
RUN --mount=type=secret,id=corp_ca \
    grep -v '^-r' requirements-dev.txt > /tmp/requirements-test.txt \
    && com-ca-corporativa pip install -r /tmp/requirements-test.txt \
    && rm /tmp/requirements-test.txt

COPY . .

# Peso baixado no build para o "pytest -m slow" rodar sem internet no container.
RUN --mount=type=secret,id=corp_ca \
    com-ca-corporativa python -c "from config import CAMINHO_MODELO; from ultralytics import YOLO; YOLO(CAMINHO_MODELO)"

CMD ["pytest"]

# ------------------------------------------------------------------------------
# ESTÁGIO 3: runtime (produção)
# ------------------------------------------------------------------------------
FROM base AS runtime

COPY config.py detectar_endpoint.py ./

# O peso .pt não vai para o Git; baixamos no build para que o container não
# precise da internet a cada cold start. Usa o mesmo caminho do config.py.
RUN --mount=type=secret,id=corp_ca \
    com-ca-corporativa python -c "from config import CAMINHO_MODELO; from ultralytics import YOLO; YOLO(CAMINHO_MODELO)"

# Usuário sem privilégios; o home é necessário porque o ultralytics grava suas
# configurações em ~/.config/Ultralytics.
RUN useradd --create-home --uid 1000 appuser
USER appuser

# HOST=0.0.0.0 expõe a API fora do container. O Render injeta PORT em tempo de
# execução e sobrescreve o valor padrão abaixo (lido por config.py).
ENV HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/', timeout=5)" || exit 1

CMD ["python", "detectar_endpoint.py"]
