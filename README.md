# API de Detecção de Objetos — YOLOv8 + FastAPI

Projeto Avaliativo do Módulo 2. A API recebe uma imagem via HTTP e devolve, em JSON,
as classes detectadas com a confiança e as coordenadas `x_min`, `y_min`, `x_max`, `y_max`.

Há dois modelos servidos lado a lado: o **base** (`yolov8n.pt`, 80 classes do COCO) e o
**ajustado** por fine-tuning (`best.pt`, dataset construction-ppe — 11 classes de EPI, das
quais 10 não existem no COCO). Enviar a mesma imagem para os dois endpoints evidencia o
ganho do treino.

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
├── postman/               # collection para testar a API publicada
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

O `best.pt`, ao contrário, **vem no clone** — ele é um artefato do treino, não existe
para download em lugar nenhum. Ou seja: quem clona já tem os dois endpoints funcionando e
só precisa rodar o treino se quiser **refazer** o modelo (outro dataset, outras épocas).
O caminho completo para isso está em [Fine-tuning](#fine-tuning).

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
dataset [**construction-ppe**](https://docs.ultralytics.com/datasets/detect/construction-ppe)
(1132 imagens de treino, 11 classes, 178 MB), baixado pelo próprio `ultralytics`.

O dataset foi escolhido porque separa os dois casos de forma limpa — o nome de cada classe
ou casa exatamente com uma do COCO, ou não casa com nenhuma:

| No COCO? | Classes                                           | O que o resultado demonstra    |
| -------- | ------------------------------------------------- | ------------------------------ |
| sim      | `person`                                          | controle: o base já detectava  |
| não      | `helmet`, `gloves`, `vest`, `boots`, `goggles`    | EPI presente — capacidade nova |
| não      | `no_helmet`, `no_gloves`, `no_boots`, `no_goggle` | EPI ausente — capacidade nova  |
| não      | `none`                                            | nenhum EPI — capacidade nova   |

As classes `no_*` marcam a *ausência* do EPI correspondente — é o que torna o dataset útil
como caso de fiscalização, e não apenas de detecção.

> Trocar o dataset **não afeta o Render**. A diferença entre uma cabeça de detecção de 4 e
> de 11 classes é de ~1.200 parâmetros (≈ 5 KB): o `best.pt` continua em ~6 MB e o consumo
> de RAM em inferência é o mesmo. O dataset em si nunca entra na imagem — ver
> [Por que o treino fica fora do deploy](#por-que-o-treino-fica-fora-do-deploy).

### Treinar no Kaggle (recomendado)

O treino exige GPU. O Kaggle oferece ~30 h por semana de graça. Use o
`notebooks/fine_tuning_kaggle.ipynb` deste repositório — ele já aponta para o
`construction-ppe`.

> A versão publicada em
> [kaggle.com/code/byronrestrepo/fine-tuning-do-yolov8-com-african-wildlife-dataset](https://www.kaggle.com/code/byronrestrepo/fine-tuning-do-yolov8-com-african-wildlife-dataset/notebook)
> ainda é a do dataset anterior. Importe o arquivo do repositório e republique, ou o
> Kaggle rodará o treino antigo.

**Passo 0 — publique o seu código.** O notebook não roda o código da sua máquina: ele
clona do GitHub. Faça commit e push antes de começar, ou o clone falha.

**Passo 1 — abra o notebook no Kaggle.** Use o link acima (`Copy & Edit`) ou importe o
arquivo: *Create → New Notebook → File → Import Notebook* e envie
`notebooks/fine_tuning_kaggle.ipynb`.

**Passo 2 — aponte o notebook para o SEU repositório.** Na célula 3, ajuste:

```python
REPO = "https://github.com/<seu-usuario>/Projeto-Avaliativo-Modulo-2.git"
BRANCH = "main"   # a branch que contém o fine_tuning.py
```

> Sem isso você treina a partir do repositório de outra pessoa. Se você clonou/forkou
> este projeto, **este passo é obrigatório**.

**Passo 3 — configure a sessão.** Os controles ficam no **editor**, não no modo de
visualização: clique em `Edit`, abra o painel lateral direito (seta `<` ou ícone ⚙) e
ajuste:

- **Accelerator → `GPU T4 x2`** (ou `P100`). Confirme o reinício da sessão.
- **Internet → `On`.** Vem desligada. Sem ela falham o `pip install`, o `git clone` e o
  download do dataset.

> Se os dois controles estiverem acinzentados, falta **verificação de telefone** na conta
> Kaggle — a plataforma exige isso para liberar GPU e internet. Há um link no próprio
> painel.

**Passo 4 — rode.** Execute a célula 1 e confirme que sai `cuda: True`; se vier `False`,
a GPU não foi aplicada. Então use **`Save Version → Save & Run All (Commit)`** (não
`Quick Save`, que não executa nada). O treino roda em segundo plano e você pode fechar a
aba. São ~15–25 min para 50 épocas numa T4.

### Treinar localmente

```bash
pip install -r requirements-train.txt
python fine_tuning.py --epocas 1 --dataset coco8.yaml   # teste de fumaça, segundos
python fine_tuning.py --epocas 50 --publicar            # treino real — LENTO sem GPU
```

`--publicar` copia o `best.pt` para a raiz do repositório, onde a API e o Dockerfile o
esperam. Sem a flag, o peso fica apenas em `runs/fine_tuning/`.

> Sem GPU, 50 épocas sobre 1132 imagens levam **horas**. Use o comando com `coco8.yaml`
> para validar que o ambiente está correto e deixe o treino real para o Kaggle.

Ao final o script imprime um relatório montado com `pandas`: um `merge(indicator=True)`
cruza as classes do modelo ajustado com as do base para separar o que é capacidade nova,
e `where`/`mask` traduzem o resultado — sem percorrer linha a linha.

> O `pandas` fica no `requirements-train.txt`, fora da imagem de produção. A API não usa
> pandas em nenhum ponto do caminho de request: são poucas caixas por imagem, e os ~50 MB
> da biblioteca pesariam num orçamento de 512 MB.

### Depois do treino: trazendo o `best.pt` para o repositório

**1. Baixe o peso.** Na versão concluída do notebook, aba **Output**, baixe o `best.pt`
que está na raiz (a última célula o copia para lá justamente para não ficar perdido entre
o dataset e a pasta `runs/`).

**2. O arquivo vem como `best.zip` — não extraia.** Um `.pt` do PyTorch *é* internamente
um ZIP, e o navegador renomeia a extensão. O arquivo baixado já é o peso completo:

```bash
mv ~/Downloads/best.zip Projeto-Avaliativo-Modulo-2/best.pt
```

> Se você extrair, vai obter `data.pkl`, `version`, `byteorder` e uma pasta `data/` —
> as vísceras do modelo, inúteis soltas. Nesse caso apague e renomeie o `.zip` original.

**3. Coloque na raiz do repositório**, no mesmo nível do `config.py`. O caminho é definido
por `CAMINHO_MODELO_AJUSTADO` no `config.py`; em qualquer outro lugar o endpoint responde
503.

**4. Confira antes de commitar:**

```bash
python -c "from ultralytics import YOLO; print(len(YOLO('best.pt').names), YOLO('best.pt').names)"
# esperado: 11 {0: 'helmet', 1: 'gloves', 2: 'vest', 3: 'boots', 4: 'goggles',
#               5: 'none', 6: 'Person', 7: 'no_helmet', 8: 'no_goggle',
#               9: 'no_gloves', 10: 'no_boots'}
```

Se aparecerem 80 classes (`person`, `bicycle`, `car`...), você pegou o `yolov8n.pt` por
engano. Se aparecerem 4 (`buffalo`, `elephant`...), é o peso do dataset antigo.

**5. Commite e faça push.** O `best.pt` é a única exceção ao `*.pt` do `.gitignore`:

```bash
git add best.pt && git commit -m "feat: peso do fine-tuning" && git push
```

O Render reconstrói ao detectar o push na branch que ele observa. Copiar o arquivo para a
pasta local sem commitar não produz efeito algum no deploy.

### O modelo ajustado não é um superconjunto do base

O fine-tuning substitui a cabeça de detecção: o `best.pt` conhece **somente** as 11 classes
do construction-ppe e perdeu as 80 do COCO. Enviar uma foto de cachorro ao `/fine_tuning/`
devolve lixo (`VEST`, `NONE`...) — ele escolhe o rótulo mais próximo que conhece, porque
"dog" não é uma opção para ele.

Isso é esperado, não é defeito. Para comparar os dois endpoints, **use uma foto de
trabalhador usando capacete e colete**:

- `/detectar/` acha só a `PERSON` — capacete e colete não existem no COCO.
- `/fine_tuning/` acha `HELMET`, `VEST` e os demais EPIs.

É aí que o ganho do treino fica visível. A classe `person` aparecer nos dois é o controle:
mostra que o ajustado não perdeu o que o base já fazia.

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
| POST   | `/fine_tuning/` | Detecta com o modelo ajustado (construction-ppe) |

Os dois endpoints POST compartilham validação, contrato de resposta e limites; mudam
apenas os pesos carregados. Se o `best.pt` não estiver presente, `/fine_tuning/` responde
**503** com instruções e o restante da API segue no ar.

### Exemplo de resposta

```json
{
  "mensagem": "Análise de imagem concluída com sucesso.",
  "modelo": "best.pt",
  "total_objetos": 2,
  "resultados": [
    {
      "classe": "HELMET",
      "confianca_pct": 92.45,
      "coordenadas": { "x_min": 48, "y_min": 112, "x_max": 310, "y_max": 428 }
    },
    {
      "classe": "VEST",
      "confianca_pct": 88.1,
      "coordenadas": { "x_min": 40, "y_min": 180, "x_max": 322, "y_max": 500 }
    }
  ]
}
```

O campo `modelo` identifica qual peso respondeu — útil ao comparar os dois endpoints
no Postman.

## Testando com o Postman

A collection fica em `postman/YOLO detect.postman_collection.json` e cobre os quatro
endpoints, com testes de contrato em cada requisição.

**Na interface:** *Import* → selecione o arquivo. Ajuste a variável `baseUrl` da collection
(`http://localhost:8000` por padrão) e, nas requisições POST, **reselecione o arquivo de
imagem** — o Postman guarda o caminho por máquina.

**Pela CLI** (já instalada neste ambiente):

```bash
postman collection run "postman/YOLO detect.postman_collection.json" --working-dir .
```

O `--working-dir .` faz o Postman resolver os caminhos relativos de `imagens/`. Para rodar
contra o deploy:

```bash
postman collection run "postman/YOLO detect.postman_collection.json" \
  --working-dir . \
  --env-var "baseUrl=https://projeto-avaliativo-modulo-2.onrender.com"
```

> O primeiro request contra o Render pode levar ~50 s: o free tier hiberna após 15 min e
> precisa subir o container. Os limites de tempo da collection já contam com isso.

## Variáveis de ambiente

| Variável               | Padrão                  | Descrição                        |
| ---------------------- | ----------------------- | -------------------------------- |
| `YOLO_MODELO`          | `yolov8n.pt`            | Peso do modelo base              |
| `YOLO_MODELO_AJUSTADO` | `best.pt`               | Peso do fine-tuning              |
| `YOLO_DATASET`         | `construction-ppe.yaml` | Dataset usado pelo treino        |
| `YOLO_CONFIANCA`       | `0.40`                  | Confiança mínima para detectar   |
| `YOLO_IMGSZ`           | `640`                   | Lado da imagem na inferência     |
| `UPLOAD_MAX_BYTES`     | `10485760`              | Tamanho máximo do upload (10 MB) |
| `HOST`                 | `127.0.0.1`             | Host do servidor                 |
| `PORT`                 | `8000`                  | Porta do servidor                |
