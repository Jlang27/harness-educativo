"""Red neuronal propia (perceptrón multicapa) que estima si una tarea exige cálculo o demostraciones.

Arquitectura (sin librerías, todo con la librería estándar):
    entrada x ∈ {0,1}^V  (bolsa de palabras + raíces de 5 letras + rasgo de símbolos matemáticos)
    capa oculta  h = tanh(W1·x + b1)          con OCULTAS neuronas
    salida       p = σ(w2·h + b2)            = P(la tarea requiere cálculo)
Se entrena con descenso del gradiente estocástico minimizando la entropía cruzada binaria.
Como x es binaria y dispersa, W1·x es solo la suma de las columnas de los rasgos presentes.
"""
import json
import math
import random
import re
import sys
import threading
import unicodedata
from pathlib import Path

OCULTAS = 8
EPOCAS = 120
TASA = 0.05
SEMILLA = 42  # entrenamiento reproducible: mismos pesos en cada arranque
SIMBOLOS = set("=+^*/∫∑√≤≥∀∃∈≠∞π")
DATOS = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "entrenamiento.json"


def normalizar(texto):
    sin_tildes = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")


def rasgos(texto):
    """Conjunto de rasgos activos: palabras y su raíz (5 letras), así 'derivada' y 'derivar' comparten 'deriv'."""
    palabras = re.findall(r"[a-z0-9]+", normalizar(texto))
    activos = {"p:" + w for w in palabras} | {"r:" + w[:5] for w in palabras if len(w) > 5}
    if any(c in SIMBOLOS for c in texto) or re.search(r"\d\s*[-+*/^=]\s*\d|\b[a-z]\s*[=^]\s*\d", texto.lower()):
        activos.add("simbolo")
    return activos


def sigmoide(s):
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, s))))


class Red:
    def __init__(self, ejemplos, semilla=SEMILLA):
        rng = random.Random(semilla)
        datos = [(rasgos(t), y) for t, y in ejemplos]
        vocabulario = sorted(set().union(*(x for x, _ in datos)))
        self.w1 = {r: [rng.uniform(-0.1, 0.1) for _ in range(OCULTAS)] for r in vocabulario}
        self.b1 = [0.0] * OCULTAS
        self.w2 = [rng.uniform(-0.5, 0.5) for _ in range(OCULTAS)]
        self.b2 = 0.0
        for _ in range(EPOCAS):
            rng.shuffle(datos)
            for x, y in datos:
                self._entrenar_paso(x, y)

    def _adelante(self, x):
        z = [self.b1[j] + sum(self.w1[r][j] for r in x) for j in range(OCULTAS)]
        h = [math.tanh(v) for v in z]
        return h, sigmoide(self.b2 + sum(w * hj for w, hj in zip(self.w2, h)))

    def _entrenar_paso(self, x, y):
        h, p = self._adelante(x)
        error = p - y  # ∂(entropía cruzada)/∂s con salida sigmoide
        for j in range(OCULTAS):
            grad_oculta = error * self.w2[j] * (1 - h[j] ** 2)  # retropropagación por tanh
            self.w2[j] -= TASA * error * h[j]
            self.b1[j] -= TASA * grad_oculta
            for r in x:
                self.w1[r][j] -= TASA * grad_oculta
        self.b2 -= TASA * error

    def probabilidad(self, texto):
        """P(la tarea requiere cálculo). Los rasgos nunca vistos en el entrenamiento se ignoran."""
        return self._adelante(rasgos(texto) & self.w1.keys())[1]


_red, _candado = None, threading.Lock()


def red():
    """La red entrenada con entrenamiento.json (se entrena una sola vez, ~1 s)."""
    global _red
    with _candado:
        if _red is None:
            ejemplos = json.loads(DATOS.read_text(encoding="utf-8"))["ejemplos"]
            _red = Red([(e["texto"], e["calculo"]) for e in ejemplos])
        return _red
