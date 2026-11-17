"""Experimento antes/después: ¿el harness responde tan bien como "siempre el modelo más potente", gastando menos?

Estrategias comparadas sobre ejercicios con respuesta conocida (experimento.json, tomados de los apuntes):
  potente  -> siempre el modelo disponible con mayor razonamiento (lo que haría una institución sin decidir)
  azar     -> un modelo disponible al azar (lo que hace quien elige sin criterio)
  harness  -> la decisión del harness (red neuronal + requisitos mínimos + arista de menor costo) con cambio de modelo

Mide: % de aciertos, cortes (preguntas sin respuesta), tokens reales cobrados por la API y costo de referencia
(tokens reales × precio de pago del modelo en OpenRouter). Resultados en resultados_experimento.csv y .json.

Uso:  python experimento.py                       (usa datos/config.json o dist/datos/config.json)
      python experimento.py --n 5 --pausa 6        (solo 5 ejercicios, 6 s entre llamadas por los cupos gratis)
"""
import argparse
import csv
import json
import random
import re
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import app
import router
from cliente import ErrorAPI

FORMATO = "\n\nTermina tu respuesta con una última línea exactamente así: RESPUESTA FINAL: <respuesta corta>"


def normalizar(texto):
    t = unicodedata.normalize("NFD", str(texto).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[*_`$\\{}]", "", t).strip(" .:\n")


def respuesta_final(texto):
    coincidencias = re.findall(r"respuesta final\s*:?\s*(.+)", normalizar(texto))
    return coincidencias[-1].strip(" .") if coincidencias else normalizar(texto).splitlines()[-1] if texto.strip() else ""


def numero(texto):
    m = re.search(r"-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?", texto)
    if not m:
        return None
    s = m.group().replace(",", ".").replace(" ", "")
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else None
    return float(s)


def es_correcta(dada, esperada):
    """Compara la respuesta final con la conocida: como número si lo es, si no como palabra."""
    dada, esperada = normalizar(dada), normalizar(esperada)
    n_esp = numero(esperada) if re.fullmatch(r"-?[\d.,/ ]+", esperada) else None
    if n_esp is not None:
        n_dada = numero(dada)
        return n_dada is not None and abs(n_dada - n_esp) < 1e-6
    return re.search(rf"\b{re.escape(esperada)}\b", dada) is not None


def preguntar_directo(clave, pregunta):
    """Una sola petición a un modelo fijo, sin respaldo. Devuelve (texto | None, entrada, salida, costo)."""
    prov, id_modelo = clave.split("/", 1)
    m = {app.clave_modelo(x): x for x in app.estado["catalogo"]}[clave]
    try:
        texto, uso = app.estado["clientes"][prov].chat(id_modelo, [{"role": "user", "content": pregunta}])
    except (ErrorAPI, OSError):
        return None, 0, 0, 0.0
    e, s = uso.get("prompt_tokens", router.tokens_estimados(pregunta)), uso.get("completion_tokens", router.tokens_estimados(texto))
    return texto, e, s, router.costo(m, e, s) or 0.0


def preguntar_harness(pregunta):
    chat = app.crear_chat({"tipo": "ejercicios", "texto": pregunta, "titulo": "experimento"})
    chat = app.enviar(chat["id"], pregunta)
    resp = [m for m in chat["mensajes"] if m["role"] == "assistant"]
    if not resp:
        return None, 0, 0, 0.0, None
    r = resp[-1]
    return r["content"], r["tokens_entrada"], r["tokens_salida"], r["costo_ref"] or 0.0, r["modelo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="cuántos ejercicios usar (por defecto todos)")
    ap.add_argument("--pausa", type=float, default=4.0, help="segundos entre llamadas (cupos gratuitos)")
    ap.add_argument("--estrategias", default="potente,azar,harness")
    ap.add_argument("--semilla", type=int, default=7)
    args = ap.parse_args()

    for ruta in (app.DATOS / "config.json", app.BASE / "dist" / "datos" / "config.json"):
        if ruta.exists():
            app.CONFIG = ruta
            break
    app.CHATS = Path(tempfile.mkdtemp())  # los chats del experimento no ensucian los del usuario
    print(f"Cargando proveedores ({app.CONFIG})…", flush=True)
    app.cargar_catalogo()
    disponibles = [m for m in app.estado["catalogo"] if m.get("disponible") and m.get("calificado", True)]
    if not disponibles:
        sys.exit("No hay modelos disponibles: revisa la conexión o las claves.")
    potente = app.clave_modelo(max(disponibles, key=lambda m: m["razonamiento"]))
    rng = random.Random(args.semilla)
    print(f"{len(disponibles)} modelos calificados disponibles. 'Potente' = {potente}\n", flush=True)

    ejercicios = json.loads((app.BASE / "experimento.json").read_text(encoding="utf-8"))["ejercicios"][: args.n]
    filas = []
    for i, ej in enumerate(ejercicios, 1):
        pregunta = ej["pregunta"] + FORMATO
        for est in args.estrategias.split(","):
            if est == "potente":
                modelo = potente
                texto, e, s, c = preguntar_directo(modelo, pregunta)
            elif est == "azar":
                modelo = app.clave_modelo(rng.choice(disponibles))
                texto, e, s, c = preguntar_directo(modelo, pregunta)
            else:
                texto, e, s, c, modelo = preguntar_harness(pregunta)
            dada = respuesta_final(texto) if texto else ""
            ok = bool(texto) and es_correcta(dada, ej["respuesta"])
            filas.append({"ejercicio": i, "estrategia": est, "modelo": modelo, "esperada": ej["respuesta"], "dada": dada[:60],
                          "correcta": ok, "corte": texto is None, "tokens_entrada": e, "tokens_salida": s, "costo_usd": c})
            print(f"[{i:2}] {est:8} {'✓' if ok else ('corte' if texto is None else '✗')}  {str(modelo)[:45]:45}  esperada={ej['respuesta']!r} dada={dada[:30]!r}", flush=True)
            time.sleep(args.pausa)

    print("\nResumen")
    print(f"{'estrategia':10} {'aciertos':>9} {'cortes':>7} {'tokens':>9} {'costo USD':>11}")
    resumen = {}
    for est in args.estrategias.split(","):
        fs = [f for f in filas if f["estrategia"] == est]
        r = {"aciertos": sum(f["correcta"] for f in fs), "total": len(fs), "cortes": sum(f["corte"] for f in fs),
             "tokens": sum(f["tokens_entrada"] + f["tokens_salida"] for f in fs), "costo_usd": sum(f["costo_usd"] for f in fs)}
        resumen[est] = r
        print(f"{est:10} {r['aciertos']:>4}/{r['total']:<4} {r['cortes']:>7} {r['tokens']:>9} {r['costo_usd']:>11.6f}")
    if "potente" in resumen and "harness" in resumen and resumen["potente"]["costo_usd"]:
        ahorro = 1 - resumen["harness"]["costo_usd"] / resumen["potente"]["costo_usd"]
        print(f"\nEl harness costó {ahorro:.0%} menos que usar siempre el modelo más potente.")

    salida = app.BASE / "resultados_experimento"
    with open(salida.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    salida.with_suffix(".json").write_text(json.dumps({"resumen": resumen, "filas": filas, "potente": potente},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Guardado en {salida.name}.csv y .json")


if __name__ == "__main__":
    main()
