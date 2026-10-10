# API de Detecção de Objetos — YOLOv8 + FastAPI

Projeto Avaliativo do Módulo 2. A API recebe uma imagem via HTTP e devolve, em JSON,
as classes detectadas com a confiança e as coordenadas `x_min`, `y_min`, `x_max`, `y_max`.

Há dois modelos servidos lado a lado: o **base** (`yolov8n.pt`, 80 classes do COCO) e o
**ajustado** por fine-tuning (`best.pt`, dataset african-wildlife). Enviar a mesma imagem
para os dois endpoints evidencia o ganho do treino.

## Estrutura

```text
Projeto-Avaliativo-Modulo-2/
├── config.py              # constantes e variáveis de ambiente
├── detectar_endpoint.py   # app FastAPI (endpoints GET e POST)
├── fine_tuning.py         # treino do modelo ajustado — NÃO roda no Render
├── requirements.txt       # dependências diretas da API
├── requirements-dev.txt   # dependências de teste
├── requirements-train.txt # dependências do treino (pandas); fora da imagem
├── notebooks/             # fine_tuning_kaggle.ipynb (treino com GPU)
├── imagens/               # imagens de teste
├── tests/                 # suíte pytest (conftest.py, dubles.py)
├── pytest.ini             # testpaths e marcador slow
├── Dockerfile             # imagem multi-stage (base, test, runtime)
├── docker-compose.yml     # atalho para subir a API e os testes localmente
├── .dockerignore          # o que não entra no contexto do build
├── yolov8n.pt             # peso base (ignorado pelo Git, baixado no build)
└── best.pt                # peso do fine-tuning (versionado: ninguém o baixa)
```

### Por que o treino fica fora do deploy

O plano free do Render oferece 0.1 vCPU e 512 MB, sem disco persistente e com
hibernação após 15 min. Isso serve inferência com folga, mas não comporta treino:
gradientes e estados do otimizador pedem alguns GB, e o serviço hiberna no meio de
qualquer job longo. O treino roda no Kaggle (GPU gratuita) e entrega um arquivo de
~6 MB; o Render apenas o serve.

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
pytest          # 27 testes com o YOLO substituído por dublê (~0,1 s)
pytest -m slow  # smoke de inferência real, carrega os pesos
```

A suíte padrão troca o modelo via `app.dependency_overrides`, então nenhum peso é
carregado. O marcador `slow` fica fora da execução normal (ver `pytest.ini`).

## Fine-tuning

O `fine_tuning.py` parte dos pesos do COCO e reaprende a cabeça de detecção sobre o
dataset **african-wildlife** (~1050 imagens, 4 classes), baixado pelo próprio
`ultralytics`. As classes foram escolhidas porque separam dois casos:

| Classe              | No COCO? | O que o resultado demonstra            |
| ------------------- | -------- | -------------------------------------- |
| `elephant`, `zebra` | sim      | o modelo base já detectava             |
| `buffalo`, `rhino`  | não      | capacidade que só existe após o treino |

### Treinar no Kaggle (recomendado)

Abra `notebooks/fine_tuning_kaggle.ipynb` no Kaggle e, em **Settings**, ligue
**Accelerator → GPU** e **Internet → ON** (vem desligada; sem ela o dataset não baixa).
O notebook clona este repositório e executa o próprio `fine_tuning.py`, então o que roda
lá é exatamente o que está versionado aqui.

### Treinar localmente

```bash
pip install -r requirements-train.txt
python fine_tuning.py --epocas 50 --publicar
python fine_tuning.py --epocas 1 --dataset coco8.yaml   # teste de fumaça, segundos
```

`--publicar` copia o `best.pt` para a raiz do repositório, onde a API e o Dockerfile o
esperam. Sem a flag, o peso fica apenas em `runs/fine_tuning/`.

Ao final o script imprime um relatório montado com `pandas`: um `merge(indicator=True)`
cruza as classes do modelo ajustado com as do base para separar o que é capacidade nova,
e `where`/`mask` traduzem o resultado — sem percorrer linha a linha.

> O `pandas` fica no `requirements-train.txt`, fora da imagem de produção. A API não usa
> pandas em nenhum ponto do caminho de request: são poucas caixas por imagem, e os ~50 MB
> da biblioteca pesariam num orçamento de 512 MB.

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

| Estágio   | Conteúdo                                                                  |
| --------- | ------------------------------------------------------------------------- |
| `base`    | Python 3.12 slim + `requirements.txt` (torch CPU no Linux)                |
| `test`    | `base` + pytest/httpx + código e testes                                   |
| `runtime` | `base` + código, o `yolov8n.pt` baixado no build e o `best.pt` versionado |

O `ultralytics` instala `opencv-python` como dependência, que exige `libGL` e conflita
com o `opencv-python-headless`. O build remove os dois e reinstala só o headless.

## Deploy no Render

1. No Render: **New → Web Service** e conecte este repositório.
2. **Language:** `Docker` · **Branch:** `main` · **Dockerfile Path:** `./Dockerfile`.
3. Não defina `PORT` nem `HOST`: o Render injeta `PORT` e o Dockerfile fixa `HOST=0.0.0.0`.
4. **Health Check Path:** `/`.
5. Depois do deploy, a documentação fica em `https://projeto-avaliativo-modulo-2.onrender.com`.

## Endpoints

| Método | Rota            | Descrição                                        |
| ------ | --------------- | ------------------------------------------------ |
| GET    | `/`             | Status da API e disponibilidade do peso ajustado |
| GET    | `/classes`      | Lista as classes que o modelo base reconhece     |
| POST   | `/detectar/`    | Detecta com o modelo base (80 classes do COCO)   |
| POST   | `/fine_tuning/` | Detecta com o modelo ajustado (african-wildlife) |

Os dois endpoints POST compartilham validação, contrato de resposta e limites; mudam
apenas os pesos carregados. Se o `best.pt` não estiver presente, `/fine_tuning/` responde
**503** com instruções e o restante da API segue no ar.

### Exemplo de resposta

```json
{
  "mensagem": "Análise de imagem concluída com sucesso.",
  "modelo": "best.pt",
  "total_objetos": 1,
  "resultados": [
    {
      "classe": "BUFFALO",
      "confianca_pct": 92.45,
      "coordenadas": { "x_min": 48, "y_min": 112, "x_max": 310, "y_max": 428 }
    }
  ]
}
```

O campo `modelo` identifica qual peso respondeu — útil ao comparar os dois endpoints
no Postman.

## Variáveis de ambiente

| Variável               | Padrão                  | Descrição                        |
| ---------------------- | ----------------------- | -------------------------------- |
| `YOLO_MODELO`          | `yolov8n.pt`            | Peso do modelo base              |
| `YOLO_MODELO_AJUSTADO` | `best.pt`               | Peso do fine-tuning              |
| `YOLO_DATASET`         | `african-wildlife.yaml` | Dataset usado pelo treino        |
| `YOLO_CONFIANCA`       | `0.40`                  | Confiança mínima para detectar   |
| `YOLO_IMGSZ`           | `640`                   | Lado da imagem na inferência     |
| `UPLOAD_MAX_BYTES`     | `10485760`              | Tamanho máximo do upload (10 MB) |
| `HOST`                 | `127.0.0.1`             | Host do servidor                 |
| `PORT`                 | `8000`                  | Porta do servidor                |
