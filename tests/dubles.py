"""Dublês que imitam a interface do ultralytics sem carregar a rede neural."""


class _Escalar:
    def __init__(self, valor):
        self._valor = valor

    def item(self):
        return self._valor


class _Caixa:
    def __init__(self, valores):
        self._valores = list(valores)

    def tolist(self):
        return self._valores


NOMES_CLASSES = {0: "person", 16: "dog"}


class BoxFalsa:
    """Mesma interface de um item de results.boxes."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [_Escalar(cls_id)]
        self.conf = [_Escalar(conf)]
        self.xyxy = [_Caixa(xyxy)]


class ResultadoFalso:
    def __init__(self, boxes, names=None):
        self.boxes = boxes
        self.names = names if names is not None else NOMES_CLASSES


class ModeloFalso:
    """Substitui o YOLO: registra a chamada de predict e devolve caixas fixas."""

    def __init__(self, boxes=None):
        self.names = NOMES_CLASSES
        self.boxes = boxes if boxes is not None else [
            BoxFalsa(cls_id=16, conf=0.9377, xyxy=[10.7, 20.2, 110.9, 220.4])
        ]
        self.chamadas = []

    def predict(self, **kwargs):
        self.chamadas.append(kwargs)
        return [ResultadoFalso(self.boxes)]
