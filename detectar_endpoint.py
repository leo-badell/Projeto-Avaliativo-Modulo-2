from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import os
import cv2
import numpy as np

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
    total_objetos: int
    resultados: list[ObjetoDetectado]


# ==============================================================================
# BLOCO 2: INICIALIZAÇÃO DO APP E DO MODELO
# ==============================================================================
app = FastAPI(
    title="API Detector de Objetos",
    description="Detecção de objetos com YOLOv8 (modelo nano) e FastAPI.",
    version="1.0.0",
)

CAMINHO_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
CONFIANCA_MINIMA = 0.40
EXTENSOES_PERMITIDAS = ("jpg", "jpeg", "png")

print("[SISTEMA] Carregando o motor YOLOv8 na memória...")
modelo = YOLO(CAMINHO_MODELO)


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


def extrair_deteccoes(resultado) -> list[ObjetoDetectado]:
    """Lê as caixas do YOLO e converte cada uma em x_min, y_min, x_max, y_max."""
    deteccoes: list[ObjetoDetectado] = []

    for box in resultado.boxes:
        cls_id = int(box.cls[0].item())
        nome_classe = modelo.names[cls_id].upper()
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
def status_servidor():
    """Endpoint GET: confirma que a API está online."""
    return {
        "status": "Online",
        "modelo": os.path.basename(CAMINHO_MODELO),
        "mensagem": "Acesse http://localhost:8000/docs para testar o endpoint /detectar/.",
    }


@app.get("/classes")
def listar_classes():
    """Endpoint GET: lista as classes que o modelo é capaz de detectar."""
    return {"total": len(modelo.names), "classes": list(modelo.names.values())}


@app.post("/detectar/", response_model=RespostaAPI)
async def detectar_objetos(arquivo: UploadFile = File(...)):
    """Endpoint POST: recebe uma imagem e devolve classes, confiança e coordenadas."""
    validar_arquivo(arquivo)

    imagem_bytes = await arquivo.read()
    imagem = bytes_para_imagem(imagem_bytes)

    resultados = modelo.predict(source=imagem, conf=CONFIANCA_MINIMA, verbose=False)
    deteccoes = extrair_deteccoes(resultados[0])

    return RespostaAPI(
        mensagem="Análise de imagem concluída com sucesso.",
        total_objetos=len(deteccoes),
        resultados=deteccoes,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
