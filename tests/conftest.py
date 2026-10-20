import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A API tem de ser importada antes do cv2, aqui como em produção: é o config,
# carregado por ela, que fixa o teto de pixels do OpenCV — e o OpenCV lê esse
# teto uma única vez. Importar o cv2 primeiro desligaria a proteção na suíte
# inteira, e os testes passariam a medir um comportamento que não é o do deploy.
import detectar_endpoint as api  # noqa: E402
from dubles import ModeloFalso  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def modelo_falso() -> ModeloFalso:
    return ModeloFalso()


@pytest.fixture
def client(modelo_falso) -> TestClient:
    """Cliente com o YOLO substituído: nenhum peso é carregado."""
    api.app.dependency_overrides[api.obter_modelo] = lambda: modelo_falso
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


@pytest.fixture
def client_ajustado(modelo_falso) -> TestClient:
    """Cliente com o modelo do fine-tuning substituído pelo dublê.

    Sem o override, obter_modelo_ajustado checa o best.pt no disco e responde
    503 — é o que o teste do peso ausente exercita.
    """
    api.app.dependency_overrides[api.obter_modelo_ajustado] = lambda: modelo_falso
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def client_real() -> TestClient:
    """Cliente com o modelo YOLO de verdade. Usado apenas nos testes marcados slow."""
    return TestClient(api.app)


@pytest.fixture
def jpeg_valido() -> bytes:
    """JPEG sintético, evita depender dos arquivos de imagens/."""
    matriz = np.full((120, 160, 3), 127, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", matriz)
    assert ok
    return buffer.tobytes()


@pytest.fixture
def png_pixels_demais() -> bytes:
    """PNG minúsculo no disco e gigante na memória: ~33 KB que viram 26 Mpx.

    É a forma do ataque que o teto de pixels barra. O limite de bytes do upload
    não enxerga diferença nenhuma entre este arquivo e uma foto comum.
    """
    matriz = np.zeros((5100, 5100), np.uint8)  # 26.01 Mpx, logo acima do teto
    ok, buffer = cv2.imencode(".png", matriz)
    assert ok
    return buffer.tobytes()
