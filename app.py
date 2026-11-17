"""Harness educativo: servidor local + ventana de la interfaz.

Al ejecutarlo (python app.py o el .exe) levanta un servidor solo en 127.0.0.1 y abre la
interfaz en una ventana de Edge en modo aplicación. Los chats y las claves se guardan
en la carpeta 'datos' junto al ejecutable.

Varios proveedores con nivel gratuito (todos con API compatible con OpenAI) se combinan en
un solo catálogo: el enrutador elige entre todos y, si uno se queda sin cuota, salta a otro.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import red_neuronal
import redes
import router
from cliente import Cliente, ErrorAPI, actualizar_catalogo, descubrir

# Con PyInstaller los archivos empaquetados están en sys._MEIPASS y los datos van junto al .exe.
CONGELADO = getattr(sys, "frozen", False)
RECURSOS = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
BASE = Path(sys.executable).parent if CONGELADO else Path(__file__).parent
DATOS = BASE / "datos"
CHATS = DATOS / "chats"
CONFIG = DATOS / "config.json"

PROVEEDORES = {
    # Cupo anónimo (sin cuenta ni clave, ~2 peticiones/min por modelo e IP): la app funciona desde el primer arranque.
    "ovh":        {"nombre": "OVHcloud",      "url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",       "clave_en": None, "sin_clave": True},
    "gemini":     {"nombre": "Google Gemini", "url": "https://generativelanguage.googleapis.com/v1beta/openai", "clave_en": "https://aistudio.google.com/apikey"},
    "nvidia":     {"nombre": "NVIDIA NIM",    "url": "https://integrate.api.nvidia.com/v1",                    "clave_en": "https://build.nvidia.com"},
    "groq":       {"nombre": "Groq",          "url": "https://api.groq.com/openai/v1",                         "clave_en": "https://console.groq.com/keys"},
    "cerebras":   {"nombre": "Cerebras",      "url": "https://api.cerebras.ai/v1",                             "clave_en": "https://cloud.cerebras.ai"},
    "mistral":    {"nombre": "Mistral",       "url": "https://api.mistral.ai/v1",                              "clave_en": "https://console.mistral.ai/api-keys"},
    "openrouter": {"nombre": "OpenRouter",    "url": "https://openrouter.ai/api/v1",                           "clave_en": "https://openrouter.ai/keys"},
    "deepseek":   {"nombre": "DeepSeek",      "url": "https://api.deepseek.com/v1",                            "clave_en": "https://platform.deepseek.com/api_keys", "pago": True},
}


def proveedores():
    """Los proveedores predefinidos más los personalizados que el usuario agregó (cualquier API compatible con OpenAI)."""
    todos = dict(PROVEEDORES)
    for p in leer_json(CONFIG, {}).get("personalizados", []):
        todos[p["id"]] = {"nombre": p["nombre"], "url": p["url"], "clave_en": None, "personalizado": True, "pago": True}
    return todos

COMUN = (" Responde en el idioma del estudiante, con un nivel adecuado para un universitario de cualquier carrera,"
         " sin jerga innecesaria. Usa Markdown y, solo si hay fórmulas, LaTeX ($...$).")
INSTRUCCION = {
    "estudiar": "Eres un tutor de cualquier materia. Explica de forma clara y por partes, con ejemplos cotidianos;"
                " si el estudiante comparte material, básate en él." + COMUN,
    "ejercicios": "Eres un tutor de cualquier materia. Haz preguntas, quizzes o tarjetas de memoria de dificultad creciente,"
                  " de a una, y corrige las respuestas del estudiante explicando el porqué. Da pistas antes que soluciones." + COMUN,
    "revision": "Eres un revisor académico de cualquier materia. Señala errores concretos (contenido, redacción, ortografía, citas,"
                " cálculos) y explica cómo corregirlos, sin reescribir el trabajo completo por el estudiante." + COMUN,
}
# Clave inválida, sin saldo, no existe, petición muy grande para el límite gratis, cuota agotada, caído
FALLAS_DE_DISPONIBILIDAD = {401, 402, 403, 404, 413, 429, 500, 502, 503}
SIN_LATIDO_MAX = 120  # segundos sin señal de la ventana antes de apagar el servidor
VENTANA = 12  # mensajes recientes que se reenvían en cada petición

estado = {"catalogo": [], "clientes": {}, "errores": {}, "cuota": None, "cargando": True, "ultimo_latido": None, "nombres": {}}


def leer_json(ruta, defecto):
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return defecto


def escribir_json(ruta, datos):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(ruta)  # escritura atómica: un cierre a mitad no corrompe el chat


def leer_claves():
    cfg = leer_json(CONFIG, {})
    claves = cfg.get("claves", {})
    if cfg.get("api_key") and "openrouter" not in claves:  # config de la versión anterior
        claves["openrouter"] = cfg["api_key"]
    return claves


def clave_modelo(m):
    return f"{m['proveedor']}/{m['id']}"


def cargar_catalogo():
    """Lee modelos.json (el de junto al .exe si existe, para poder editarlo) y lo cruza con cada proveedor."""
    estado["cargando"] = True
    externo = BASE / "modelos.json"
    ruta = externo if externo.exists() else RECURSOS / "modelos.json"
    catalogo = leer_json(ruta, {"modelos": []})["modelos"]
    claves = leer_claves()
    clientes, errores, cuota = {}, {}, None
    for prov, info in proveedores().items():
        if not claves.get(prov) and not info.get("sin_clave") and not info.get("personalizado"):
            continue  # un personalizado puede no pedir clave (p. ej. Ollama local)
        cliente = Cliente(info["url"], claves.get(prov, ""))
        try:
            vivos = cliente.modelos_en_vivo()
        except (ErrorAPI, OSError, KeyError, ValueError) as e:
            errores[prov] = str(e)[:200]
            continue
        clientes[prov] = cliente
        actualizar_catalogo([m for m in catalogo if m["proveedor"] == prov], vivos, usar_precios=not info.get("sin_clave"))
        # El banco crece solo: todo modelo de chat que el proveedor ofrezca entra (sin calificar si no está en modelos.json).
        # De OpenRouter solo los ':free', porque su lista completa son cientos de modelos de pago.
        descubrir(catalogo, prov, vivos, gratis=not info.get("pago"), solo_free=prov == "openrouter")
        if prov == "openrouter":
            cuota = cliente.cuota()
    for m in catalogo:
        if m["proveedor"] not in clientes:
            m["disponible"] = False
    red_neuronal.red()  # entrena la red aquí (hilo de arranque) para que el primer chat no espere
    estado.update(catalogo=catalogo, clientes=clientes, errores=errores, cuota=cuota, cargando=False,
                  nombres={p: i["nombre"] for p, i in proveedores().items()})


def recomendar_clave(catalogo, tipo, tokens, calculo, peso_actual):
    """Si un proveedor sin configurar tiene un modelo mejor que el elegido, lo sugiere.

    Se repite la elección suponiendo que esos proveedores estuvieran conectados (como si estuvieran disponibles);
    si el ganador hipotético es de uno de ellos y pesa más, vale la pena conseguir esa clave gratuita.
    """
    claves = leer_claves()
    faltan = {p for p, i in PROVEEDORES.items() if not i.get("sin_clave") and not i.get("pago") and not claves.get(p)}
    hipotetico = [dict(m, disponible=True) if m["proveedor"] in faltan else m for m in catalogo if m["gratis"]]
    ranking, _ = router.elegir(hipotetico, tipo, tokens, calculo)
    if not ranking:
        return None
    w, m = max(ranking, key=lambda par: par[0])  # aquí interesa la calidad, no el costo: es una clave gratuita
    if m["proveedor"] in faltan and w > (peso_actual or 0):
        return dict(ficha(m, w), clave_en=PROVEEDORES[m["proveedor"]]["clave_en"])
    return None


def ruta_chat(chat_id):
    if not all(c in "0123456789abcdef" for c in chat_id):  # evita rutas tipo ../../
        raise ValueError("id inválido")
    return CHATS / f"{chat_id}.json"


def ficha(m, peso=None, costo=None):
    """Datos reales del modelo que la interfaz usa para justificar la elección."""
    return {"clave": clave_modelo(m), "id": m["id"], "proveedor": estado["nombres"].get(m["proveedor"], m["proveedor"]),
            "contexto": m["contexto"], "razonamiento": m["razonamiento"], "velocidad": m["velocidad"], "peso": peso,
            "gratis": m["gratis"], "calificado": m.get("calificado", True), "fuente": m.get("fuente"),
            "precio_entrada": m.get("precio_entrada"), "precio_salida": m.get("precio_salida"), "costo": costo}


def ventana(mensajes, k=VENTANA):
    """Lo que se envía a la API: instrucciones y material siempre, más solo los últimos k mensajes del chat.

    Sin ventana, el mensaje j reenvía los j anteriores y el total de tokens tras n mensajes es t·n(n+1)/2
    (crece cuadrático, ver inducción en el documento); con ventana cada envío cuesta a lo más k·t (lineal).
    shortcut: los mensajes viejos se descartan sin resumir; resumirlos con un modelo barato conservaría más contexto.
    """
    fijos = [m for m in mensajes if m["role"] == "system" or m.get("oculto")]
    charla = [m for m in mensajes if m["role"] in ("user", "assistant") and not m.get("oculto")]
    return [{"role": m["role"], "content": m["content"]} for m in fijos + charla[-k:]]


def crear_chat(d):
    tipo = d.get("tipo") if d.get("tipo") in router.PESOS else "estudiar"
    material = str(d.get("material") or "")
    texto = str(d.get("texto") or "")
    tokens = router.tokens_estimados(material + texto) + router.tokens_estimados(INSTRUCCION[tipo])
    # Se exige rigor matemático si el estudiante lo pide o si la red neuronal detecta cálculo.
    p_calculo = red_neuronal.red().probabilidad(texto + "\n" + material[:3000])
    interruptor = bool(d.get("calculo"))
    calculo = interruptor or p_calculo >= 0.5
    catalogo = estado["catalogo"]
    # Los modelos de pago solo están disponibles si el usuario puso su propia clave: eso es su consentimiento a pagarlos.
    ranking, descartados = router.elegir(catalogo, tipo, tokens, calculo)
    cadena = [clave_modelo(m) for _, m in ranking]
    # Unidad 4: red de sustitución entre los aptos; el respaldo sigue las capas de BFS desde el elegido.
    adj = redes.construir([(clave_modelo(m), m["id"], m["razonamiento"]) for _, m in ranking])
    red = None
    if cadena:
        cadena = [cadena[0]] + redes.orden_respaldo(adj, cadena[0], {k: i for i, k in enumerate(cadena)})
        cb = redes.intermediacion(adj, normalizada=True)
        puente = max(cb, key=cb.get)
        vistos, componentes = set(), 0
        for v in adj:  # cada BFS desde un vértice no visitado descubre una componente conexa
            if v not in vistos:
                componentes += 1
                vistos |= set(redes.bfs(adj, v))
        red = {"nodos": len(adj), "aristas": sum(len(v) for v in adj.values()) // 2, "componentes": componentes,
               "vecinos": len(adj[cadena[0]]), "grado": round(redes.grado(adj, cadena[0]), 3),
               "cercania": round(redes.cercania(adj, cadena[0]), 3), "puente": puente, "intermediacion": round(cb[puente], 3)}
    # Elección manual: ese modelo va primero y los recomendados quedan de respaldo.
    manual = next((m for m in catalogo if clave_modelo(m) == d.get("modelo")), None)
    if manual:
        cadena = [clave_modelo(manual)] + [k for k in cadena if k != clave_modelo(manual)]
    chat = {
        "id": uuid.uuid4().hex,
        "titulo": (d.get("titulo") or "Nuevo chat")[:60],
        "creado": time.time(),
        "tipo": tipo,
        "material_nombre": d.get("material_nombre") or "",
        "decision": {
            "tokens": tokens, "calculo": calculo, "interruptor": interruptor, "p_calculo": round(p_calculo, 3),
            "pesos": router.PESOS[tipo], "umbral_calculo": router.UMBRAL_CALCULO, "umbral_general": router.UMBRAL_GENERAL,
            "salida": router.SALIDA[tipo], "red": red,
            "recomendacion": recomendar_clave(catalogo, tipo, tokens, calculo, max((w for w, _ in ranking), default=0)),
            "manual": ficha(manual) if manual else None,
            "aptos": [ficha(m, w, router.costo(m, tokens, router.SALIDA[tipo])) for w, m in ranking],
            "descartados": [dict(ficha(m), falla=f) for m, f in descartados],
        },
        "cadena": cadena,
        "mensajes": [{"role": "system", "content": INSTRUCCION[tipo]}]
        + ([{"role": "user", "content": "Material de trabajo:\n" + material, "oculto": True}] if material else []),
    }
    escribir_json(ruta_chat(chat["id"]), chat)
    return chat


def enviar(chat_id, texto):
    chat = leer_json(ruta_chat(chat_id), None)
    if chat is None:
        raise LookupError("El chat no existe")
    if not estado["clientes"]:
        raise PermissionError("Configura al menos un proveedor en Ajustes para poder chatear.")
    chat["mensajes"].append({"role": "user", "content": texto})
    para_api = ventana(chat["mensajes"])
    tokens = sum(router.tokens_estimados(m["content"]) for m in para_api)
    por_clave = {clave_modelo(m): m for m in estado["catalogo"]}

    def saltar(motivo):
        modelo = chat["cadena"].pop(0).split("/", 1)[1]
        siguiente = f" Sigo con {chat['cadena'][0].split('/', 1)[1]} para que no se corte tu sesión." if chat["cadena"] else ""
        chat["mensajes"].append({"role": "aviso", "content": f"⚡ {modelo} {motivo}.{siguiente}"})

    while chat["cadena"]:
        modelo = chat["cadena"][0]
        prov, id_modelo = modelo.split("/", 1)
        cliente, ficha_m = estado["clientes"].get(prov), por_clave.get(modelo, {})
        # X(m) se vuelve a evaluar en cada mensaje: el chat crece y puede dejar de caber en el respaldo.
        if ficha_m.get("contexto", 10**9) < tokens * 1.2:
            saltar(f"no alcanza a leer una conversación tan larga (≈{tokens} tokens)")
            continue
        try:
            if cliente is None:
                raise ErrorAPI(401, "proveedor sin clave")
            respuesta, uso = cliente.chat(id_modelo, para_api)
            entrada, salida = uso.get("prompt_tokens", tokens), uso.get("completion_tokens", router.tokens_estimados(respuesta))
            chat["mensajes"].append({"role": "assistant", "content": respuesta, "modelo": modelo,
                                     "tokens_entrada": entrada, "tokens_salida": salida, "medido": bool(uso),
                                     "costo_ref": router.costo(ficha_m, entrada, salida) if ficha_m else None})
            break
        except ErrorAPI as e:
            if e.codigo not in FALLAS_DE_DISPONIBILIDAD:
                chat["mensajes"].append({"role": "error", "content": str(e)})
                break
            saltar({429: "agotó su cupo por ahora", 503: "está saturado", 401: "no tiene una clave válida", 403: "no tiene una clave válida",
                    404: "ya no está disponible", 413: "no acepta un mensaje tan largo"}.get(e.codigo, "no respondió") + f" (código {e.codigo})")
        except OSError:
            # Sin respuesta a tiempo (modelo saturado o razonando demasiado): se trata igual que un 503.
            saltar("tardó demasiado en responder")
    else:
        chat["mensajes"].append({"role": "error", "content": "Ningún modelo recomendado está disponible ahora. Intenta más tarde o configura otro proveedor."})
    escribir_json(ruta_chat(chat_id), chat)
    return chat


def listar_chats():
    chats = [leer_json(p, None) for p in CHATS.glob("*.json")]
    chats = [c for c in chats if c]
    chats.sort(key=lambda c: c["creado"], reverse=True)
    return [{"id": c["id"], "titulo": c["titulo"], "tipo": c["tipo"]} for c in chats]


def resumen_estado():
    claves = leer_claves()
    return {
        "cargando": estado["cargando"], "cuota": estado["cuota"],
        "proveedores": [{"id": p, "nombre": i["nombre"], "clave_en": i["clave_en"], "tiene_clave": bool(claves.get(p)), "sin_clave": bool(i.get("sin_clave")),
                         "personalizado": bool(i.get("personalizado")), "pago": bool(i.get("pago")), "url": i["url"],
                         "conectado": p in estado["clientes"], "error": estado["errores"].get(p)} for p, i in proveedores().items()],
        "modelos": [dict(ficha(m), disponible=m.get("disponible")) for m in estado["catalogo"]],
    }


class Manejador(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # sin ruido en consola

    def responder(self, codigo, cuerpo, tipo="application/json; charset=utf-8"):
        datos = cuerpo if isinstance(cuerpo, bytes) else json.dumps(cuerpo, ensure_ascii=False).encode()
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def origen_valido(self):
        # Solo la propia interfaz puede usar la API; bloquea páginas web ajenas que apunten a localhost.
        origen = self.headers.get("Origin")
        return origen is None or origen == f"http://{self.headers.get('Host')}"

    def cuerpo(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def despachar(self, metodo):
        if not self.origen_valido():
            return self.responder(403, {"error": "origen no permitido"})
        partes = [p for p in self.path.split("?")[0].split("/") if p]
        try:
            if metodo == "GET" and not partes:
                return self.responder(200, (RECURSOS / "index.html").read_bytes(), "text/html; charset=utf-8")
            if partes[:1] != ["api"]:
                return self.responder(404, {"error": "no encontrado"})
            ruta = partes[1:]
            if ruta == ["ping"]:
                estado["ultimo_latido"] = time.time()
                return self.responder(200, {"ok": True})
            if ruta == ["estado"] and metodo == "GET":
                return self.responder(200, resumen_estado())
            if ruta == ["config"] and metodo == "POST":
                # {"claves": {"groq": "gsk_..."}}: cadena vacía = no tocar, null = quitar la clave
                # {"personalizado": {"nombre", "url", "clave"}} agrega un proveedor; {"quitar_personalizado": id} lo borra
                d, cfg, claves = self.cuerpo(), leer_json(CONFIG, {}), leer_claves()
                pers = cfg.get("personalizados", [])
                if d.get("personalizado"):
                    nuevo = d["personalizado"]
                    nombre, url = str(nuevo.get("nombre", "")).strip()[:40], str(nuevo.get("url", "")).strip().rstrip("/")
                    if not nombre or not url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
                        raise PermissionError("El proveedor necesita un nombre y una URL https:// (o local).")
                    pid = "pers-" + "".join(c if c.isalnum() else "-" for c in nombre.lower())
                    pers = [p for p in pers if p["id"] != pid] + [{"id": pid, "nombre": nombre, "url": url}]
                    if str(nuevo.get("clave", "")).strip():
                        claves[pid] = str(nuevo["clave"]).strip()
                if d.get("quitar_personalizado"):
                    pers = [p for p in pers if p["id"] != d["quitar_personalizado"]]
                    claves.pop(d["quitar_personalizado"], None)
                cfg["personalizados"] = pers
                validos = set(PROVEEDORES) | {p["id"] for p in pers}
                for prov, valor in (d.get("claves") or {}).items():
                    if prov not in validos:
                        continue
                    if valor is None:
                        claves.pop(prov, None)
                    elif str(valor).strip():
                        claves[prov] = str(valor).strip()
                cfg.pop("api_key", None)
                cfg["claves"] = claves
                escribir_json(CONFIG, cfg)
                cargar_catalogo()
                return self.responder(200, resumen_estado())
            if ruta == ["chats"] and metodo == "GET":
                return self.responder(200, listar_chats())
            if ruta == ["chats"] and metodo == "POST":
                return self.responder(200, crear_chat(self.cuerpo()))
            if len(ruta) == 2 and ruta[0] == "chats":
                if metodo == "GET":
                    chat = leer_json(ruta_chat(ruta[1]), None)
                    return self.responder(200, chat) if chat else self.responder(404, {"error": "El chat no existe"})
                if metodo == "DELETE":
                    ruta_chat(ruta[1]).unlink(missing_ok=True)
                    return self.responder(200, {"ok": True})
            if len(ruta) == 3 and ruta[0] == "chats" and ruta[2] == "mensaje" and metodo == "POST":
                return self.responder(200, enviar(ruta[1], str(self.cuerpo().get("texto", ""))))
            return self.responder(404, {"error": "no encontrado"})
        except PermissionError as e:
            return self.responder(400, {"error": str(e)})
        except (LookupError, ValueError) as e:
            return self.responder(404, {"error": str(e)})

    def do_GET(self):
        self.despachar("GET")

    def do_POST(self):
        self.despachar("POST")

    def do_DELETE(self):
        self.despachar("DELETE")


def abrir_ventana(url):
    """Edge en modo app (ventana sin barra de navegador); si no está, el navegador por defecto."""
    rutas = [shutil.which("msedge"),
             os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
             os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe")]
    for edge in filter(None, rutas):
        if Path(edge).exists():
            subprocess.Popen([edge, f"--app={url}", "--window-size=1200,800",
                              # perfil propio fuera de OneDrive: ventana separada del Edge del usuario
                              f"--user-data-dir={Path(os.environ.get('LOCALAPPDATA', DATOS)) / 'HarnessEducativo' / 'ventana'}"])
            return
    webbrowser.open(url)


def vigilar_cierre(servidor):
    # Cuando la ventana se cierra deja de mandar latidos: apagamos para no dejar el proceso colgado.
    while True:
        time.sleep(10)
        ultimo = estado["ultimo_latido"]
        if ultimo and time.time() - ultimo > SIN_LATIDO_MAX:
            servidor.shutdown()
            return


def main():
    CHATS.mkdir(parents=True, exist_ok=True)
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), Manejador)  # puerto libre al azar
    url = f"http://127.0.0.1:{servidor.server_address[1]}/"
    print(f"Harness educativo en {url}  (cierra la ventana para salir)")
    # Consultar a los proveedores tarda; la ventana se abre ya y muestra "Conectando…" mientras tanto.
    threading.Thread(target=cargar_catalogo, daemon=True).start()
    threading.Thread(target=vigilar_cierre, args=(servidor,), daemon=True).start()
    abrir_ventana(url)
    servidor.serve_forever()


if __name__ == "__main__":
    main()
