"""Casos de resultado conocido para el enrutador. Ejecutar: python -m unittest -v"""
import unittest

import router

# Catálogo fijo de prueba (no depende de la red ni de modelos.json).
A = {"id": "A-rapido-tonto",  "contexto": 32_000,    "gratis": True,  "razonamiento": 0.3,  "velocidad": 0.95}
B = {"id": "B-razonador",     "contexto": 160_000,   "gratis": True,  "razonamiento": 0.9,  "velocidad": 0.3}
C = {"id": "C-contexto-1M",   "contexto": 1_048_576, "gratis": True,  "razonamiento": 0.7,  "velocidad": 0.9}
D = {"id": "D-pago-top",      "contexto": 200_000,   "gratis": False, "razonamiento": 0.95, "velocidad": 0.6}
CATALOGO = [A, B, C, D]


def ids(ranking):
    return [m["id"] for _, m in ranking]


class TestRouter(unittest.TestCase):
    def test_calculo_excluye_modelos_debiles(self):
        ranking, _ = router.elegir([A, B, C], "ejercicios", calculo=True)
        self.assertEqual(ids(ranking), ["B-razonador"])  # único con razonamiento >= 0.8

    def test_material_largo_exige_contexto(self):
        ranking, descartados = router.elegir(CATALOGO, "estudiar", tokens=200_000)
        self.assertEqual(ids(ranking), ["C-contexto-1M"])  # 200k * 1.5 = 300k solo cabe en C
        self.assertIn("contexto suficiente", dict((m["id"], f) for m, f in descartados)["B-razonador"])

    def test_razonamiento_bajo_se_descarta(self):
        _, descartados = router.elegir(CATALOGO, "revision")
        fallas = {m["id"]: f for m, f in descartados}
        self.assertEqual(fallas["A-rapido-tonto"], ["razonamiento suficiente"])  # 0.3 < UMBRAL_GENERAL

    def test_elige_el_apto_mas_barato(self):
        caro = dict(B, id="caro", precio_entrada=1.5, precio_salida=9.0)
        barato = dict(C, id="barato", razonamiento=0.81, precio_entrada=0.04, precio_salida=0.08)
        ranking, _ = router.elegir([caro, barato], "ejercicios", calculo=True)
        self.assertEqual(ids(ranking), ["barato", "caro"])  # ambos aptos: gana el de menor costo
        # costo a mano: (0 tokens de entrada) + 600 de salida * 0.08 USD / 1M
        self.assertAlmostEqual(router.costo(barato, 0, 600), 600 * 0.08 / 1_000_000)

    def test_sin_precio_va_al_final(self):
        con_precio = dict(B, id="con", precio_entrada=5, precio_salida=5)
        ranking, _ = router.elegir([dict(D, id="sin"), con_precio], "revision")
        self.assertEqual(ids(ranking), ["con", "sin"])

    def test_revision_prefiere_razonamiento(self):
        ranking, _ = router.elegir(CATALOGO, "revision")
        self.assertEqual(ids(ranking)[0], "D-pago-top")

    def test_no_disponible_se_descarta(self):
        caido = dict(B, disponible=False)
        ranking, _ = router.elegir([A, caido], "ejercicios", calculo=True)
        self.assertEqual(ranking, [])  # sin modelo apto: el programa lo dice en vez de usar uno malo

    def test_peso_a_mano(self):
        # w(ejercicios, A) = 0.1*(32000/1048576) + 0.4*0.3 + 0.5*0.95
        esperado = 0.1 * (32_000 / 1_048_576) + 0.4 * 0.3 + 0.5 * 0.95
        self.assertAlmostEqual(router.peso("ejercicios", A), esperado)


if __name__ == "__main__":
    unittest.main()
