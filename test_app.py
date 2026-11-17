"""Prueba la cadena de respaldo contra una API falsa local: el 1.º modelo responde 429 y el 2.º contesta."""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import app
from cliente import Cliente


class APIFalsa(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        modelo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"]
        codigo, cuerpo = (429, {"error": "cuota"}) if modelo == "agotado" else \
            (200, {"choices": [{"message": {"content": f"hola desde {modelo}"}}]})
        datos = json.dumps(cuerpo).encode()
        self.send_response(codigo)
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)


class TestRespaldo(unittest.TestCase):
    def test_salta_al_siguiente_modelo_si_hay_429(self):
        servidor = ThreadingHTTPServer(("127.0.0.1", 0), APIFalsa)
        threading.Thread(target=servidor.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as tmp:
            app.CHATS = Path(tmp)
            app.estado["catalogo"] = [
                {"proveedor": "falso", "id": "agotado", "contexto": 100_000, "gratis": True, "razonamiento": 0.9, "velocidad": 0.9},
                {"proveedor": "falso", "id": "respaldo", "contexto": 100_000, "gratis": True, "razonamiento": 0.9, "velocidad": 0.5},
                {"proveedor": "sin-clave", "id": "otro", "contexto": 100_000, "gratis": True, "razonamiento": 0.9, "velocidad": 0.1},
            ]
            app.estado["clientes"] = {"falso": Cliente(f"http://127.0.0.1:{servidor.server_address[1]}", "clave-falsa")}

            chat = app.crear_chat({"tipo": "ejercicios"})
            self.assertEqual(chat["cadena"], ["falso/agotado", "falso/respaldo", "sin-clave/otro"])  # ordenado por peso
            chat = app.enviar(chat["id"], "hola")
            self.assertEqual(chat["mensajes"][-1]["content"], "hola desde respaldo")
            self.assertEqual(chat["mensajes"][-1]["modelo"], "falso/respaldo")
            self.assertEqual(chat["cadena"], ["falso/respaldo", "sin-clave/otro"])

            # Elección manual: el modelo elegido va primero; si falla, sigue la cadena recomendada.
            chat = app.crear_chat({"tipo": "ejercicios", "modelo": "sin-clave/otro"})
            self.assertEqual(chat["cadena"][0], "sin-clave/otro")
            chat = app.enviar(chat["id"], "hola")  # 'sin-clave' no tiene cliente -> salta; 'agotado' da 429 -> salta
            self.assertEqual(chat["mensajes"][-1]["modelo"], "falso/respaldo")
        servidor.shutdown()


class TestVentana(unittest.TestCase):
    def test_tokens_crecen_lineal_con_ventana(self):
        # Caso de resultado conocido (inducción): sin ventana se envían t·n(n+1)/2 mensajes en total.
        n, k = 40, app.VENTANA
        mensajes = [{"role": "system", "content": "s"}]
        sin, con = 0, 0
        for j in range(1, n + 1):
            mensajes.append({"role": "user", "content": "x"})
            sin += sum(1 for m in mensajes if m["role"] == "user")
            con += sum(1 for m in app.ventana(mensajes) if m["role"] == "user")
        self.assertEqual(sin, n * (n + 1) // 2)  # 820
        self.assertEqual(con, k * (k + 1) // 2 + (n - k) * k)  # crece de a k por mensaje: lineal
        self.assertEqual(app.ventana(mensajes)[0]["role"], "system")  # las instrucciones nunca se pierden


class TestDescubrir(unittest.TestCase):
    def test_agrega_solo_modelos_de_chat_nuevos(self):
        from cliente import descubrir
        catalogo = [{"proveedor": "g", "id": "conocido", "contexto": 1000, "gratis": True, "razonamiento": 0.9, "velocidad": 0.5}]
        vivos = {"models/conocido": {}, "models/nuevo-chat": {"context_window": 64000},
                 "models/gemini-embedding-2": {}, "models/algo-tts": {}}
        nuevos = descubrir(catalogo, "g", vivos, gratis=True)
        self.assertEqual([m["id"] for m in nuevos], ["nuevo-chat"])  # embeddings y TTS no son de chat
        self.assertEqual(nuevos[0]["contexto"], 64000)
        self.assertLess(nuevos[0]["razonamiento"], 0.8)  # sin calificar: nunca pasa el filtro de cálculo

    def test_openrouter_solo_free(self):
        from cliente import descubrir
        nuevos = descubrir([], "or", {"a/b:free": {}, "a/c": {}}, gratis=True, solo_free=True)
        self.assertEqual([m["id"] for m in nuevos], ["a/b:free"])


if __name__ == "__main__":
    unittest.main()
