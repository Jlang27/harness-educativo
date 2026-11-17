"""Enrutador: decide qué modelo de IA atiende cada tarea, al menor costo que cumpla los requisitos.

Modelo matemático (lo que se sustenta):
  1. Requisitos mínimos -> un modelo es apto si cumple los tres:
     disponible (aparece en la API y no agotó cuota), su contexto alcanza para el material y
     su razonamiento es >= UMBRAL_CALCULO si la tarea es de cálculo, o >= UMBRAL_GENERAL si no.
  2. Grafo bipartito ponderado -> tipos de tarea (izquierda) vs modelos (derecha). Cada arista (t, m)
     de un modelo apto lleva dos pesos: el costo estimado de atender la tarea con m y la calidad
         w(t,m) = Σ PESOS[t][k] · atributo_k(m).
     Se elige la arista de MENOR COSTO; si dos cuestan lo mismo, la de mayor calidad. El resto,
     en ese mismo orden, es la cadena de respaldo si el elegido falla (cuota, caída, timeout).
"""
import math

UMBRAL_CALCULO = 0.80     # GPQA mínimo para cálculo y demostraciones: evita modelos que se equivocan en matemáticas
UMBRAL_GENERAL = 0.60     # GPQA mínimo para cualquier tarea: debajo de esto el modelo no sirve para estudiar
MARGEN_CONTEXTO = 1.5     # el material + la conversación necesitan más que solo el texto pegado
CONTEXTO_MAX = 1_048_576  # para normalizar el contexto a [0, 1]

# Peso de cada atributo según el tipo de tarea (cada fila suma 1).
PESOS = {
    "estudiar":  {"contexto": 0.5, "razonamiento": 0.3, "velocidad": 0.2},  # mucho material
    "ejercicios": {"contexto": 0.1, "razonamiento": 0.4, "velocidad": 0.5},  # practicar / memorizar
    "revision":  {"contexto": 0.1, "razonamiento": 0.8, "velocidad": 0.1},  # revisar trabajos (con 0.2 de contexto, un modelo de 1M ganaba a uno que razona mejor)
}
# Tokens de respuesta típicos por tarea, para estimar el costo antes de preguntar.
SALIDA = {"estudiar": 800, "ejercicios": 600, "revision": 600}


def tokens_estimados(texto):
    # shortcut: ~4 caracteres por token en español; usar un tokenizador real si se necesita precisión
    return len(texto) // 4


def requisitos(m, tokens, calculo):
    """Qué requisitos cumple el modelo, para poder explicar por qué se descartó."""
    return {
        "disponible": m.get("disponible", True),
        "contexto suficiente": m["contexto"] >= tokens * MARGEN_CONTEXTO,
        "razonamiento suficiente": m["razonamiento"] >= (UMBRAL_CALCULO if calculo else UMBRAL_GENERAL),
    }


def peso(tipo, m):
    attrs = {
        "contexto": min(m["contexto"] / CONTEXTO_MAX, 1.0),
        "razonamiento": m["razonamiento"],
        "velocidad": m["velocidad"],
    }
    return sum(p * attrs[k] for k, p in PESOS[tipo].items())


def costo(m, tokens_entrada, tokens_salida):
    """USD que cuesta una petición con el precio de referencia del modelo; None si no se conoce."""
    if m.get("precio_entrada") is None or m.get("precio_salida") is None:
        return None
    return (tokens_entrada * m["precio_entrada"] + tokens_salida * m["precio_salida"]) / 1_000_000


def elegir(modelos, tipo, tokens=0, calculo=False):
    """Devuelve (ranking, descartados).

    ranking: [(peso, modelo)] de los aptos, del más barato al más caro (empates: mayor peso primero)
             -> [0] es el elegido, el resto es respaldo. Un modelo sin precio conocido va al final.
    descartados: [(modelo, [requisitos que no cumple])] para mostrar por qué no se usó cada uno.
    """
    ranking, descartados = [], []
    for m in modelos:
        falsas = [nombre for nombre, v in requisitos(m, tokens, calculo).items() if not v]
        if falsas:
            descartados.append((m, falsas))
        else:
            ranking.append((round(peso(tipo, m), 3), m))

    def clave(par):
        c = costo(par[1], tokens, SALIDA[tipo])
        return (math.inf if c is None else c, -par[0])

    ranking.sort(key=clave)
    return ranking, descartados
