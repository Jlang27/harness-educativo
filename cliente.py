"""Cliente HTTP para cualquier API compatible con OpenAI (por defecto OpenRouter). Solo librería estándar."""
import json
import urllib.error
import urllib.request


class ErrorAPI(Exception):
    def __init__(self, codigo, mensaje):
        super().__init__(f"HTTP {codigo}: {mensaje}")
        self.codigo = codigo


class Cliente:
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _pedir(self, ruta, cuerpo=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:  # los proveedores sin clave (cupo anónimo) no llevan Authorization
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + ruta,
            data=json.dumps(cuerpo).encode() if cuerpo is not None else None,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise ErrorAPI(e.code, e.read().decode(errors="replace")[:300]) from None

    def modelos_en_vivo(self):
        """{id: modelo} según la API. En OpenRouter trae context_length y precios (0 = gratis)."""
        return {m["id"]: m for m in self._pedir("/models")["data"]}

    def cuota(self):
        """Estado de la clave (OpenRouter: /key). None si el proveedor no lo expone."""
        try:
            return self._pedir("/key").get("data")
        except ErrorAPI:
            return None

    def chat(self, modelo, mensajes):
        """(respuesta, uso). 'uso' trae los tokens que la API realmente cobró: prompt_tokens y completion_tokens."""
        r = self._pedir("/chat/completions", {"model": modelo, "messages": mensajes})
        return r["choices"][0]["message"]["content"], r.get("usage") or {}


NO_CHAT = ("embed", "tts", "image", "guard", "safety", "transcribe", "clip", "video", "audio", "robotics", "live",
           "rerank", "translate", "detector", "calibration", "retriever", "starcoder", "cosmos", "topic-control",
           "whisper", "nvr-", "bge", "computer-use", "nano-banana", "decision")


def contexto_vivo(vivo):
    # OpenRouter usa context_length, Groq context_window, vLLM max_model_len; los demás no lo informan
    c = vivo.get("context_length") or vivo.get("context_window") or vivo.get("max_model_len")
    return c if isinstance(c, int) and c > 0 else None


def descubrir(catalogo, proveedor, en_vivo, gratis, solo_free=False):
    """Agrega al catálogo los modelos de chat que el proveedor ofrece y no estaban calificados.

    Entran con razonamiento 0.5 ('sin calificar'): quedan por debajo del umbral de cálculo hasta que
    alguien los califique con un benchmark en modelos.json.
    """
    conocidos = {m["id"] for m in catalogo if m["proveedor"] == proveedor}
    nuevos = []
    for crudo, vivo in en_vivo.items():
        id_ = crudo.removeprefix("models/")
        if id_ in conocidos or any(p in id_.lower() for p in NO_CHAT) or (solo_free and not id_.endswith(":free")):
            continue
        nuevos.append({"proveedor": proveedor, "id": id_, "contexto": contexto_vivo(vivo) or 32768, "gratis": gratis,
                       "razonamiento": 0.5, "velocidad": 0.5, "calificado": False, "disponible": True})
    catalogo.extend(nuevos)
    return nuevos


def actualizar_catalogo(catalogo, en_vivo, usar_precios=True):
    """Cruza el catálogo local con lo que la API reporta: disponibilidad, contexto real y gratuidad.

    usar_precios=False para proveedores que publican precios aunque el cupo usado sea gratis (OVH anónimo).
    """
    en_vivo = {k.removeprefix("models/"): v for k, v in en_vivo.items()}  # Gemini lista "models/<id>"
    for m in catalogo:
        vivo = en_vivo.get(m["id"])
        m["disponible"] = vivo is not None
        if vivo:
            m["contexto"] = contexto_vivo(vivo) or m["contexto"]
            precios = vivo.get("pricing")
            if precios and usar_precios:
                m["gratis"] = float(precios.get("prompt", 1)) == 0 and float(precios.get("completion", 1)) == 0
    return catalogo
