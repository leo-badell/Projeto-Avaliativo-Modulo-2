import os
import sys

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detectar_endpoint as api
from dubles import ModeloFalso


@pytest.fixture
def modelo_falso() -> ModeloFalso:
    return ModeloFalso()


@pytest.fixture
def client(modelo_falso) -> TestClient:
    """Cliente com o YOLO substituído: nenhum peso é carregado."""
    api.app.dependency_overrides[api.obter_modelo] = lambda: modelo_falso
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
