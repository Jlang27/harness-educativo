"""Red de sustitución entre modelos (Unidad 4, análisis de redes).

Modelación (las cuatro preguntas de la Unidad 4):
  Nodos:     los modelos aptos para la pregunta actual.
  Aristas:   {u, v} si u y v son intercambiables: el mismo modelo base en otro proveedor, o con
             razonamiento casi igual (|R(u) - R(v)| <= TOLERANCIA).
  Dirección: no dirigida (si u sustituye a v, v sustituye a u).
  Peso:      no ponderada; las distancias se miden en saltos con BFS.

Uso: si el modelo elegido falla, BFS desde él da el orden de respaldo: primero sus sustitutos
directos (capa 1), luego los de capa 2, y así; dentro de una capa, el más barato. La centralidad
de grado del elegido dice cuántos respaldos equivalentes tiene; la intermediación señala qué modelo
conecta grupos de calidad distintos.
Representación: lista de adyacencia {nodo: set(vecinos)} (Unidad 3, 3.3).
"""
from collections import deque
from itertools import combinations

TOLERANCIA = 0.05


def base(id_modelo):
    """'google/gemma-4-31b-it:free' y 'gemma-4-31b-it' son el mismo modelo base: 'gemma-4-31b'."""
    b = id_modelo.split("/")[-1].lower().removesuffix(":free")
    for sufijo in ("-it", "-instruct"):
        b = b.removesuffix(sufijo)
    return b


def construir(modelos):
    """modelos: [(clave, id, razonamiento)] -> lista de adyacencia."""
    adj = {c: set() for c, _, _ in modelos}
    for (c1, i1, r1), (c2, i2, r2) in combinations(modelos, 2):
        if base(i1) == base(i2) or abs(r1 - r2) <= TOLERANCIA:
            adj[c1].add(c2)
            adj[c2].add(c1)
    return adj


def bfs(adj, origen):
    """Distancias desde origen (número de aristas); los no alcanzables no aparecen (d = ∞)."""
    dist, cola = {origen: 0}, deque([origen])
    while cola:
        u = cola.popleft()
        for v in sorted(adj[u]):  # orden alfabético, como en clase
            if v not in dist:
                dist[v] = dist[u] + 1
                cola.append(v)
    return dist


def grado(adj, v):
    """C_D(v) = deg(v) / (n - 1)."""
    n = len(adj)
    return len(adj[v]) / (n - 1) if n > 1 else 0.0


def cercania(adj, v):
    """Variante armónica H(v) = 1/(n-1) Σ 1/d(v,u), con 1/∞ = 0: sirve aunque la red no sea conexa (Unidad 4, 4.3)."""
    n = len(adj)
    return sum(1 / d for u, d in bfs(adj, v).items() if u != v) / (n - 1) if n > 1 else 0.0


def intermediacion(adj, normalizada=False):
    """C_B(v) = Σ_{s<t} σ_st(v)/σ_st, con el algoritmo de Brandes (un BFS por vértice)."""
    cb = dict.fromkeys(adj, 0.0)
    for s in adj:
        pila, pred = [], {v: [] for v in adj}
        sigma, dist = dict.fromkeys(adj, 0), {s: 0}
        sigma[s], cola = 1, deque([s])
        while cola:
            u = cola.popleft()
            pila.append(u)
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    cola.append(w)
                if dist[w] == dist[u] + 1:  # u está en un camino más corto hacia w
                    sigma[w] += sigma[u]
                    pred[w].append(u)
        delta = dict.fromkeys(adj, 0.0)
        while pila:
            w = pila.pop()
            for u in pred[w]:
                delta[u] += sigma[u] / sigma[w] * (1 + delta[w])
            if w != s:
                cb[w] += delta[w]
    n = len(adj)
    cb = {v: c / 2 for v, c in cb.items()}  # no dirigido: cada par {s,t} se contó dos veces
    if normalizada and n > 2:
        cb = {v: c / ((n - 1) * (n - 2) / 2) for v, c in cb.items()}
    return cb


def orden_respaldo(adj, elegido, prioridad):
    """Cadena de respaldo: por capas de BFS desde el elegido; en cada capa, menor 'prioridad' primero.

    Los modelos no alcanzables (otra componente conexa) van al final, también por prioridad.
    """
    dist = bfs(adj, elegido)
    resto = [v for v in adj if v != elegido]
    return sorted(resto, key=lambda v: (dist.get(v, float("inf")), prioridad[v]))
