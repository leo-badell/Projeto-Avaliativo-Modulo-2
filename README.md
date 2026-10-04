# API de Detecção de Objetos — YOLOv8 + FastAPI

Projeto Avaliativo do Módulo 2. A API recebe uma imagem via HTTP e devolve, em JSON,
as classes detectadas com a confiança e as coordenadas `x_min`, `y_min`, `x_max`, `y_max`.

## Estrutura

```text
Projeto-Avaliativo-Modulo-2/
├── config.py              # constantes e variáveis de ambiente
├── detectar_endpoint.py   # app FastAPI (endpoints GET e POST)
├── requirements.txt       # dependências diretas
├── requirements-dev.txt   # dependências de teste
├── imagens/               # imagens de teste
├── tests/                 # suíte pytest (conftest.py, dubles.py)
├── pytest.ini             # testpaths e marcador slow
├── Dockerfile             # imagem multi-stage (base, test, runtime)
├── docker-compose.yml     # atalho para subir a API e os testes localmente
├── .dockerignore          # o que não entra no contexto do build
└── yolov8n.pt             # peso do modelo (ignorado pelo Git)
```

## Como rodar

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows (bash) — no CMD use .venv\Scripts\activate
pip install -r requirements.txt
uvicorn detectar_endpoint:app --reload
```

Documentação interativa em `http://localhost:8000/docs`.

O arquivo `yolov8n.pt` não é versionado. Se ele não existir, o `ultralytics` faz o
download automático na primeira execução.

## Testes

```bash
pip install -r requirements-dev.txt
pytest          # 22 testes com o YOLO substituído por dublê (~0,1 s)
pytest -m slow  # smoke de inferência real, carrega os pesos
```

A suíte padrão troca o modelo via `app.dependency_overrides`, então nenhum peso é
carregado. O marcador `slow` fica fora da execução normal (ver `pytest.ini`).

## Rodando com Docker

```bash
docker build -t detector-api .                      # imagem de produção
docker run --rm -p 8000:8000 detector-api           # http://localhost:8000/docs

docker compose up --build                           # o mesmo, via compose

docker build --target test -t detector-api-test .   # imagem com pytest
docker run --rm detector-api-test                   # roda a suíte no container
docker compose --profile test run --rm tests        # o mesmo, via compose
```

O `Dockerfile` tem três estágios:

| Estágio   | Conteúdo                                                                 |
| --------- | ------------------------------------------------------------------------ |
| `base`    | Python 3.12 slim + `requirements.txt` (torch CPU no Linux)               |
| `test`    | `base` + pytest/httpx + código e testes                                  |
| `runtime` | `base` + `config.py`, `detectar_endpoint.py` e o `yolov8n.pt` baixado no build |

O `ultralytics` instala `opencv-python` como dependência, que exige `libGL` e conflita
com o `opencv-python-headless`. O build remove os dois e reinstala só o headless.

## Deploy no Render

1. No Render: **New → Web Service** e conecte este repositório.
2. **Language:** `Docker` · **Branch:** `main` · **Dockerfile Path:** `./Dockerfile`.
3. Não defina `PORT` nem `HOST`: o Render injeta `PORT` e o Dockerfile fixa `HOST=0.0.0.0`.
4. **Health Check Path:** `/`.
5. Depois do deploy, a documentação fica em `https://<seu-servico>.onrender.com/docs`.

## Endpoints

| Método | Rota         | Descrição                                |
| ------ | ------------ | ---------------------------------------- |
| GET    | `/`          | Status da API e modelo carregado         |
| GET    | `/classes`   | Lista as classes que o modelo reconhece  |
| POST   | `/detectar/` | Recebe uma imagem e retorna as detecções |

### Exemplo de resposta

```json
{
  "mensagem": "Análise de imagem concluída com sucesso.",
  "total_objetos": 1,
  "resultados": [
    {
      "classe": "DOG",
      "confianca_pct": 92.45,
      "coordenadas": { "x_min": 48, "y_min": 112, "x_max": 310, "y_max": 428 }
    }
  ]
}
```

## Variáveis de ambiente

| Variável           | Padrão       | Descrição                        |
| ------------------ | ------------ | -------------------------------- |
| `YOLO_MODELO`      | `yolov8n.pt` | Peso do modelo a carregar        |
| `YOLO_CONFIANCA`   | `0.40`       | Confiança mínima para detectar   |
| `YOLO_IMGSZ`       | `640`        | Lado da imagem na inferência     |
| `UPLOAD_MAX_BYTES` | `10485760`   | Tamanho máximo do upload (10 MB) |
| `HOST`             | `127.0.0.1`  | Host do servidor                 |
| `PORT`             | `8000`       | Porta do servidor                |
