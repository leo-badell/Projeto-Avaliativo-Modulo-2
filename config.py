"""Configurações centrais da API de detecção de objetos."""

import os

RAIZ_PROJETO = os.path.dirname(os.path.abspath(__file__))

# O peso .pt não vai para o Git; se não existir, o ultralytics baixa no primeiro uso.
NOME_MODELO = os.getenv("YOLO_MODELO", "yolov8n.pt")
CAMINHO_MODELO = os.path.join(RAIZ_PROJETO, NOME_MODELO)

# Peso do fine-tuning (endpoint /fine_tuning/). Diferente do base, este arquivo é
# commitado: ninguém o baixa de lugar nenhum, ele é produzido pelo fine_tuning.py.
# Se estiver ausente, a API continua de pé e só o /fine_tuning/ responde 503.
NOME_MODELO_AJUSTADO = os.getenv("YOLO_MODELO_AJUSTADO", "best.pt")
CAMINHO_MODELO_AJUSTADO = os.path.join(RAIZ_PROJETO, NOME_MODELO_AJUSTADO)

# Dataset usado pelo fine_tuning.py; o ultralytics baixa sozinho pelo nome.
DATASET_FINE_TUNING = os.getenv("YOLO_DATASET", "african-wildlife.yaml")

CONFIANCA_MINIMA = float(os.getenv("YOLO_CONFIANCA", "0.40"))
# Lado da imagem usado na inferência; reduzir acelera a rede e perde objetos pequenos.
TAMANHO_INFERENCIA = int(os.getenv("YOLO_IMGSZ", "640"))
EXTENSOES_PERMITIDAS = ("jpg", "jpeg", "png")
TAMANHO_MAXIMO_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", 10 * 1024 * 1024))

HOST = os.getenv("HOST", "127.0.0.1")
PORTA = int(os.getenv("PORT", "8000"))
