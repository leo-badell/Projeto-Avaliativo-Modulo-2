import os

import pytest
from fastapi import HTTPException, UploadFile

import detectar_endpoint as api
from config import (
    CAMINHO_MODELO,
    CAMINHO_MODELO_AJUSTADO,
    TAMANHO_MAXIMO_BYTES,
    TAMANHO_INFERENCIA,
    CONFIANCA_MINIMA,
)
from dubles import BoxFalsa, ResultadoFalso


def _upload(nome: str) -> UploadFile:
    arquivo = UploadFile.__new__(UploadFile)
    arquivo.filename = nome
    return arquivo


# ==============================================================================
# UNITÁRIOS: bytes_para_imagem
# ==============================================================================
def test_bytes_para_imagem_decodifica_matriz_bgr(jpeg_valido):
    imagem = api.bytes_para_imagem(jpeg_valido)
    assert imagem.shape == (120, 160, 3)


def test_bytes_para_imagem_rejeita_bytes_invalidos():
    with pytest.raises(HTTPException) as erro:
        api.bytes_para_imagem(b"isto nao e uma imagem")
    assert erro.value.status_code == 400


# ==============================================================================
# UNITÁRIOS: validar_arquivo
# ==============================================================================
@pytest.mark.parametrize("nome", ["foto.jpg", "foto.jpeg", "foto.png", "FOTO.JPG"])
def test_validar_arquivo_aceita_extensoes_suportadas(nome):
    api.validar_arquivo(_upload(nome))


@pytest.mark.parametrize("nome", ["doc.pdf", "script.exe", "arquivo_sem_extensao", ""])
def test_validar_arquivo_rejeita_demais_formatos(nome):
    with pytest.raises(HTTPException) as erro:
        api.validar_arquivo(_upload(nome))
    assert erro.value.status_code == 415


# ==============================================================================
# UNITÁRIOS: extrair_deteccoes (conversão das coordenadas)
# ==============================================================================
def test_extrair_deteccoes_converte_xyxy_em_coordenadas():
    resultado = ResultadoFalso([BoxFalsa(cls_id=16, conf=0.9377, xyxy=[10.7, 20.2, 110.9, 220.4])])

    deteccoes = api.extrair_deteccoes(resultado)

    assert len(deteccoes) == 1
    coords = deteccoes[0].coordenadas
    assert (coords.x_min, coords.y_min, coords.x_max, coords.y_max) == (10, 20, 110, 220)
    assert coords.x_min < coords.x_max and coords.y_min < coords.y_max


def test_extrair_deteccoes_converte_confianca_em_percentual():
    resultado = ResultadoFalso([BoxFalsa(cls_id=0, conf=0.5, xyxy=[0, 0, 1, 1])])

    assert api.extrair_deteccoes(resultado)[0].confianca_pct == 50.0


def test_extrair_deteccoes_usa_nome_da_classe_em_maiusculo():
    resultado = ResultadoFalso([BoxFalsa(cls_id=0, conf=0.8, xyxy=[0, 0, 1, 1])])

    assert api.extrair_deteccoes(resultado)[0].classe == "PERSON"


def test_extrair_deteccoes_sem_caixas_retorna_lista_vazia():
    assert api.extrair_deteccoes(ResultadoFalso([])) == []


# ==============================================================================
# INTEGRAÇÃO: endpoints (YOLO substituído por dublê)
# ==============================================================================
def test_get_raiz_retorna_status_online(client):
    resposta = client.get("/")

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "Online"


def test_get_classes_usa_o_mapa_de_classes_do_modelo(client):
    corpo = client.get("/classes").json()

    assert corpo["total"] == len(corpo["classes"])
    assert "dog" in corpo["classes"]


def test_post_detectar_serializa_a_deteccao_no_contrato(client, jpeg_valido):
    resposta = client.post(
        "/detectar/", files={"arquivo": ("teste.jpg", jpeg_valido, "image/jpeg")}
    )

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"mensagem", "modelo", "total_objetos", "resultados"}
    assert corpo["total_objetos"] == len(corpo["resultados"]) == 1
    assert corpo["resultados"][0] == {
        "classe": "DOG",
        "confianca_pct": 93.77,
        "coordenadas": {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 220},
    }


def test_post_detectar_repassa_os_parametros_de_config(client, modelo_falso, jpeg_valido):
    client.post("/detectar/", files={"arquivo": ("teste.jpg", jpeg_valido, "image/jpeg")})

    chamada = modelo_falso.chamadas[0]
    assert chamada["conf"] == CONFIANCA_MINIMA
    assert chamada["imgsz"] == TAMANHO_INFERENCIA


def test_post_detectar_rejeita_arquivo_nao_imagem(client, modelo_falso):
    resposta = client.post("/detectar/", files={"arquivo": ("doc.pdf", b"%PDF-1.4", "application/pdf")})

    assert resposta.status_code == 415
    assert modelo_falso.chamadas == []


def test_post_detectar_rejeita_jpg_corrompido(client, modelo_falso):
    resposta = client.post(
        "/detectar/", files={"arquivo": ("falso.jpg", b"conteudo invalido", "image/jpeg")}
    )

    assert resposta.status_code == 400
    assert modelo_falso.chamadas == []


def test_post_detectar_rejeita_upload_acima_do_limite(client, modelo_falso):
    gigante = b"\xff" * (TAMANHO_MAXIMO_BYTES + 1)

    resposta = client.post("/detectar/", files={"arquivo": ("grande.jpg", gigante, "image/jpeg")})

    assert resposta.status_code == 413
    assert modelo_falso.chamadas == []


def test_post_detectar_exige_o_campo_arquivo(client):
    assert client.post("/detectar/").status_code == 422


# ==============================================================================
# INTEGRAÇÃO: endpoint /fine_tuning/
# ==============================================================================
def test_post_fine_tuning_responde_no_mesmo_contrato(client_ajustado, jpeg_valido):
    resposta = client_ajustado.post(
        "/fine_tuning/", files={"arquivo": ("teste.jpg", jpeg_valido, "image/jpeg")}
    )

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"mensagem", "modelo", "total_objetos", "resultados"}
    assert corpo["modelo"] == os.path.basename(CAMINHO_MODELO_AJUSTADO)


def test_post_fine_tuning_aplica_as_mesmas_validacoes(client_ajustado, modelo_falso):
    resposta = client_ajustado.post(
        "/fine_tuning/", files={"arquivo": ("doc.pdf", b"%PDF-1.4", "application/pdf")}
    )

    assert resposta.status_code == 415
    assert modelo_falso.chamadas == []


def test_post_detectar_identifica_o_modelo_base(client, jpeg_valido):
    resposta = client.post("/detectar/", files={"arquivo": ("t.jpg", jpeg_valido, "image/jpeg")})

    assert resposta.json()["modelo"] == os.path.basename(CAMINHO_MODELO)


def test_obter_modelo_ajustado_responde_503_sem_o_peso(monkeypatch):
    """Sem o best.pt a API segue de pé; só este endpoint fica indisponível."""
    monkeypatch.setattr(api.os.path, "exists", lambda _: False)

    with pytest.raises(HTTPException) as erro:
        api.obter_modelo_ajustado()

    assert erro.value.status_code == 503
    assert "fine_tuning.py" in erro.value.detail


def test_carregar_modelo_cacheia_por_caminho():
    """lru_cache(maxsize=2) comporta os dois pesos sem recarregar a cada request."""
    assert api.carregar_modelo.cache_info().maxsize == 2


# ==============================================================================
# SMOKE COM O MODELO REAL: rode com `pytest -m slow`
# ==============================================================================
@pytest.mark.slow
def test_inferencia_real_detecta_cachorro(client_real):
    with open("imagens/dogs.jpeg", "rb") as arquivo:
        resposta = client_real.post(
            "/detectar/", files={"arquivo": ("dogs.jpeg", arquivo.read(), "image/jpeg")}
        )

    assert resposta.status_code == 200
    classes = {item["classe"] for item in resposta.json()["resultados"]}
    assert "DOG" in classes
