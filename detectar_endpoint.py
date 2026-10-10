from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from ultralytics import YOLO
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any
import os
import cv2
import numpy as np

from config import (
    CAMINHO_MODELO,
    CAMINHO_MODELO_AJUSTADO,
    CONFIANCA_MINIMA,
    TAMANHO_INFERENCIA,
    EXTENSOES_PERMITIDAS,
    TAMANHO_MAXIMO_BYTES,
    HOST,
    PORTA,
)

# ==============================================================================
# BLOCO 1: MOLDES PYDANTIC (FORMATO DA RESPOSTA JSON)
# ==============================================================================
class Coordenadas(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class ObjetoDetectado(BaseModel):
    classe: str
    confianca_pct: float
    coordenadas: Coordenadas


class RespostaAPI(BaseModel):
    mensagem: str
    modelo: str  # identifica qual peso respondeu: base ou ajustado
    total_objetos: int
    resultados: list[ObjetoDetectado]


class RespostaClasses(BaseModel):
    total: int
    classes: list[str]


# ==============================================================================
# BLOCO 2: INICIALIZAÇÃO DO APP E DO MODELO
# ==============================================================================
@lru_cache(maxsize=2)
def carregar_modelo(caminho: str) -> YOLO:
    """Carrega um peso uma única vez. maxsize=2: o base e o do fine-tuning."""
    print(f"[SISTEMA] Carregando {os.path.basename(caminho)}")
    return YOLO(caminho)


def obter_modelo() -> YOLO:
    """Modelo base (COCO). Nos testes é substituído via dependency_overrides."""
    return carregar_modelo(CAMINHO_MODELO)


def obter_modelo_ajustado() -> YOLO:
    """Modelo do fine-tuning.

    O peso é um artefato do treino, não um download: se ninguém rodou o
    fine_tuning.py ainda, o arquivo não existe. Responder 503 aqui mantém o
    resto da API no ar em vez de derrubar o processo no boot.
    """
    if not os.path.exists(CAMINHO_MODELO_AJUSTADO):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Modelo ajustado ausente ({os.path.basename(CAMINHO_MODELO_AJUSTADO)}). "
                "Rode 'python fine_tuning.py --publicar' e faça o commit do peso."
            ),
        )
    return carregar_modelo(CAMINHO_MODELO_AJUSTADO)


@asynccontextmanager
async def ciclo_de_vida(_app: FastAPI):
    # Aquece só o modelo base. O ajustado carrega sob demanda, na primeira
    # chamada ao /fine_tuning/: em 512 MB, pagar as duas cargas no boot aperta
    # demais a memória e o cold start do free tier.
    obter_modelo()
    yield


app = FastAPI(
    title="API Detector de Objetos",
    description="Detecção de objetos com YOLOv8 (modelo nano) e FastAPI.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


# ==============================================================================
# BLOCO 3: FUNÇÕES AUXILIARES
# ==============================================================================
def bytes_para_imagem(imagem_bytes: bytes) -> np.ndarray:
    """Converte os bytes recebidos na requisição em uma matriz de imagem do OpenCV."""
    vetor = np.frombuffer(imagem_bytes, np.uint8)
    imagem = cv2.imdecode(vetor, cv2.IMREAD_COLOR)
    if imagem is None:
        raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem.")
    return imagem


def extrair_deteccoes(resultado: Any) -> list[ObjetoDetectado]:
    """Lê as caixas do YOLO e converte cada uma em x_min, y_min, x_max, y_max."""
    # Any: aceita tanto o Results do ultralytics quanto o ResultadoFalso dos testes
    deteccoes: list[ObjetoDetectado] = []

    for box in resultado.boxes:
        cls_id = int(box.cls[0].item())
        nome_classe = str(resultado.names[cls_id]).upper()
        confianca = round(float(box.conf[0].item()) * 100, 2)

        # box.xyxy traz o canto superior esquerdo e o inferior direito (Pascal VOC)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        deteccoes.append(
            ObjetoDetectado(
                classe=nome_classe,
                confianca_pct=confianca,
                coordenadas=Coordenadas(
                    x_min=int(x1),
                    y_min=int(y1),
                    x_max=int(x2),
                    y_max=int(y2),
                ),
            )
        )

    return deteccoes


def validar_arquivo(arquivo: UploadFile) -> None:
    """Bloqueia arquivos que não sejam imagens suportadas."""
    extensao = (arquivo.filename or "").split(".")[-1].lower()
    if extensao not in EXTENSOES_PERMITIDAS:
        raise HTTPException(status_code=415, detail="Envie uma imagem jpg, jpeg ou png.")


# ==============================================================================
# BLOCO 4: ENDPOINTS GET E POST
# ==============================================================================
@app.get("/")
def status_servidor() -> dict[str, str | bool]:
    """Endpoint GET: confirma que a API está online."""
    return {
        "status": "Online",
        "modelo": os.path.basename(CAMINHO_MODELO),
        # Expor isto evita adivinhação depois do deploy: se vier false, o peso
        # do fine-tuning não entrou na imagem.
        "modelo_ajustado": os.path.basename(CAMINHO_MODELO_AJUSTADO),
        "modelo_ajustado_disponivel": os.path.exists(CAMINHO_MODELO_AJUSTADO),
        "mensagem": "Acesse /docs para testar os endpoints /detectar/ e /fine_tuning/.",
    }


@app.get("/classes", response_model=RespostaClasses)
def listar_classes(modelo: YOLO = Depends(obter_modelo)) -> RespostaClasses:
    """Endpoint GET: lista as classes que o modelo é capaz de detectar."""
    classes = [str(nome) for nome in modelo.names.values()]
    return RespostaClasses(total=len(classes), classes=classes)


async def analisar_upload(arquivo: UploadFile, modelo: YOLO, rotulo_modelo: str) -> RespostaAPI:
    """Pipeline comum aos dois endpoints de detecção.

    A única diferença entre /detectar/ e /fine_tuning/ são os pesos carregados;
    validação, decodificação e extração das caixas são idênticas.
    """
    validar_arquivo(arquivo)

    imagem_bytes = await arquivo.read()
    if len(imagem_bytes) > TAMANHO_MAXIMO_BYTES:
        raise HTTPException(status_code=413, detail="Imagem maior que o limite permitido.")

    imagem = bytes_para_imagem(imagem_bytes)

    # list(): o ultralytics tipa o retorno como Iterator | list; o ignore cobre o
    # parâmetro sem tipo da própria assinatura do predict (código da biblioteca).
    resultados = list(
        modelo.predict(  # pyright: ignore[reportUnknownMemberType]
            source=imagem,
            conf=CONFIANCA_MINIMA,
            imgsz=TAMANHO_INFERENCIA,
            verbose=False,
        )
    )
    deteccoes = extrair_deteccoes(resultados[0])

    return RespostaAPI(
        mensagem="Análise de imagem concluída com sucesso.",
        modelo=rotulo_modelo,
        total_objetos=len(deteccoes),
        resultados=deteccoes,
    )


@app.post("/detectar/", response_model=RespostaAPI)
async def detectar_objetos(
    arquivo: UploadFile = File(...),
    modelo: YOLO = Depends(obter_modelo),
):
    """Endpoint POST: detecta com o modelo base (80 classes do COCO)."""
    return await analisar_upload(arquivo, modelo, os.path.basename(CAMINHO_MODELO))


@app.post("/fine_tuning/", response_model=RespostaAPI)
async def detectar_objetos_ajustado(
    arquivo: UploadFile = File(...),
    modelo: YOLO = Depends(obter_modelo_ajustado),
):
    """Endpoint POST: detecta com o modelo do fine-tuning (african-wildlife).

    Mesma imagem enviada aqui e no /detectar/ evidencia o ganho do treino:
    buffalo e rhino não existem no COCO, então só este endpoint os encontra.
    """
    return await analisar_upload(arquivo, modelo, os.path.basename(CAMINHO_MODELO_AJUSTADO))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORTA)
