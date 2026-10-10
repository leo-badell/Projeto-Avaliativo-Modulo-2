"""Fine-tuning do YOLOv8 sobre o dataset african-wildlife.

Este script NÃO roda no Render: o plano free (0.1 vCPU / 512 MB) não comporta os
gradientes e o otimizador do treino. Ele roda com GPU no Kaggle (ou na CPU local,
para um teste curto) e produz um único artefato — o best.pt — que é a única coisa
que viaja para o deploy.

O dataset african-wildlife é baixado pelo próprio ultralytics; não há nada para
preparar à mão. Suas quatro classes dividem o problema em dois casos que o
relatório final evidencia:

    elephant, zebra  -> já existem no COCO (o modelo base já detectava)
    buffalo,  rhino  -> não existem no COCO (capacidade genuinamente nova)

Uso:
    python fine_tuning.py --epocas 50 --publicar
    python fine_tuning.py --epocas 1 --dataset coco8.yaml   # fumaça, em segundos
"""

# A API de treino do ultralytics (train, val, trainer.best, box.class_result) não é
# tipada: no modo estrito cada acesso vira um erro de "tipo parcialmente desconhecido".
# Silenciar as regras no arquivo inteiro é mais legível que repetir ~18 ignores
# inline — e só vale aqui, onde todo o contato com a biblioteca está concentrado.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from config import (
    CAMINHO_MODELO,
    CAMINHO_MODELO_AJUSTADO,
    DATASET_FINE_TUNING,
    RAIZ_PROJETO,
    TAMANHO_INFERENCIA,
)

# O ultralytics escreve os artefatos em <projeto>/<nome>/weights/best.pt.
# O caminho precisa ser absoluto: com um relativo, o ultralytics o concatena ao
# próprio runs_dir e o resultado aninha (runs/detect/runs/fine_tuning/...).
PROJETO_SAIDA = str(Path(RAIZ_PROJETO) / "runs" / "fine_tuning")
NOME_EXECUCAO = "african_wildlife"

# Abaixo deste mAP50 a classe entra no relatório marcada para revisão.
LIMIAR_REVISAO = 0.50


# ==============================================================================
# BLOCO 1: TREINO
# ==============================================================================
def treinar(
    dataset: str,
    epocas: int,
    imgsz: int,
    batch: int,
    paciencia: int,
    modelo_base: str,
) -> tuple[YOLO, Path]:
    """Roda as épocas de fine-tuning e devolve o modelo treinado e o best.pt.

    Partimos de pesos já treinados no COCO em vez de inicialização aleatória: as
    camadas de extração de features são reaproveitadas e só a cabeça de detecção
    precisa realmente aprender as classes novas. É o que torna viável treinar com
    ~1000 imagens em vez de centenas de milhares.
    """
    modelo = YOLO(modelo_base)

    modelo.train(
        data=dataset,
        epochs=epocas,
        imgsz=imgsz,
        batch=batch,
        patience=paciencia,  # para se o val estagnar: poupa quota de GPU
        project=PROJETO_SAIDA,
        name=NOME_EXECUCAO,
        exist_ok=True,
        seed=42,  # reprodutibilidade: a mesma chamada gera o mesmo resultado
        plots=True,
    )

    return modelo, Path(modelo.trainer.best)


# ==============================================================================
# BLOCO 2: MÉTRICAS E RELATÓRIO (pandas)
# ==============================================================================
def metricas_por_classe(modelo: YOLO, dataset: str, imgsz: int) -> pd.DataFrame:
    """Roda a validação e tabula precisão, recall e mAP de cada classe.

    O ultralytics entrega as métricas como arrays paralelos indexados por
    ap_class_index. Montar um DataFrame a partir deles evita carregar esses
    índices pelo resto do código.
    """
    resultado = modelo.val(data=dataset, imgsz=imgsz, verbose=False)
    caixa = resultado.box

    # class_result(i) devolve (precisão, recall, mAP50, mAP50-95) da i-ésima classe.
    linhas = [
        (str(resultado.names[indice]).lower(), *caixa.class_result(i))
        for i, indice in enumerate(caixa.ap_class_index)
    ]

    return (
        pd.DataFrame(linhas, columns=["classe", "precisao", "recall", "map50", "map50_95"])
        .round(4)
        .sort_values("classe", ignore_index=True)
    )


def mapa_de_capacidades(classes_ajustado: list[str], classes_base: list[str]) -> pd.DataFrame:
    """Cruza as classes do modelo ajustado com as do modelo base (COCO).

    Um merge com indicator resolve a classificação sem nenhum if/else: o próprio
    _merge já diz se a classe existia no modelo base ("both") ou se é nova
    ("left_only"). O where() apenas traduz esse marcador para texto legível.
    """
    ajustado = pd.DataFrame({"classe": pd.Series(classes_ajustado, dtype="string").str.lower()})
    base = pd.DataFrame({"classe": pd.Series(classes_base, dtype="string").str.lower()})

    cruzamento = ajustado.merge(base, on="classe", how="left", indicator=True)
    ja_existia = cruzamento["_merge"].eq("both")

    return cruzamento.assign(
        capacidade=pd.Series("CAPACIDADE NOVA", index=cruzamento.index).where(
            ~ja_existia, "ja existia no COCO"
        )
    ).drop(columns="_merge")


def relatorio(metricas: pd.DataFrame, capacidades: pd.DataFrame) -> pd.DataFrame:
    """Junta métricas e capacidades na tabela que vai para a apresentação.

    O merge é left a partir das capacidades, então uma classe do modelo sem
    nenhuma instância no conjunto de validação entra como NaN. fillna(0) a
    trata como desempenho zero, que é o que o alerta precisa sinalizar.
    """
    colunas_metricas = ["precisao", "recall", "map50", "map50_95"]

    return (
        capacidades.merge(metricas, on="classe", how="left")
        .fillna(dict.fromkeys(colunas_metricas, 0.0))
        .assign(
            # mask() marca o que ficou abaixo do aceitável sem percorrer linha a linha.
            alerta=lambda df: pd.Series("", index=df.index).mask(
                df["map50"] < LIMIAR_REVISAO, "<- revisar"
            )
        )
        .sort_values(["capacidade", "map50"], ascending=[True, False], ignore_index=True)
    )


# ==============================================================================
# BLOCO 3: PUBLICAÇÃO DO ARTEFATO
# ==============================================================================
def publicar(peso: Path, destino: Path) -> Path:
    """Copia o best.pt para a raiz do repositório, onde a API e o Docker o esperam."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(peso, destino)
    return destino


# ==============================================================================
# BLOCO 4: CLI
# ==============================================================================
def analisar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default=DATASET_FINE_TUNING, help="yaml do dataset (padrao: %(default)s)")
    parser.add_argument("--epocas", type=int, default=50, help="numero de epocas (padrao: %(default)s)")
    parser.add_argument("--imgsz", type=int, default=TAMANHO_INFERENCIA, help="lado da imagem (padrao: %(default)s)")
    parser.add_argument("--batch", type=int, default=16, help="imagens por passo; reduza se faltar VRAM")
    parser.add_argument("--paciencia", type=int, default=10, help="epocas sem melhora antes de parar")
    parser.add_argument("--modelo-base", default=CAMINHO_MODELO, help="pesos de partida")
    parser.add_argument("--publicar", action="store_true", help="copia o best.pt para a raiz do repositorio")
    return parser.parse_args()


def main() -> None:
    args = analisar_argumentos()

    modelo, peso = treinar(
        dataset=args.dataset,
        epocas=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        paciencia=args.paciencia,
        modelo_base=args.modelo_base,
    )

    metricas = metricas_por_classe(modelo, args.dataset, args.imgsz)
    capacidades = mapa_de_capacidades(
        classes_ajustado=[str(nome) for nome in modelo.names.values()],
        classes_base=[str(nome) for nome in YOLO(args.modelo_base).names.values()],
    )
    tabela = relatorio(metricas, capacidades)

    print("\n" + "=" * 78)
    print("RESULTADO DO FINE-TUNING")
    print("=" * 78)
    print(tabela.to_string(index=False))
    print(f"\nmAP50 medio: {metricas['map50'].mean():.4f}")
    print(f"peso treinado: {peso}")

    if args.publicar:
        destino = publicar(peso, Path(CAMINHO_MODELO_AJUSTADO))
        print(f"publicado em: {destino}  <- commitar este arquivo")
    else:
        print("\n(use --publicar para copiar o best.pt para a raiz do repositorio)")


if __name__ == "__main__":
    main()
