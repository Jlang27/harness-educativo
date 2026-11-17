"""Casos de resultado conocido: el grafo R de los apuntes de la Unidad 4 (diapositivas 9 a 18)."""
import unittest

import redes

# R: A-B, A-C, B-D, C-D, D-E, E-F
R = {"A": {"B", "C"}, "B": {"A", "D"}, "C": {"A", "D"}, "D": {"B", "C", "E"}, "E": {"D", "F"}, "F": {"E"}}


class TestGrafoR(unittest.TestCase):
    def test_bfs_distancias(self):
        self.assertEqual(redes.bfs(R, "A"), {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3, "F": 4})

    def test_lema_apreton_de_manos(self):
        self.assertEqual(sum(len(v) for v in R.values()), 2 * 6)  # Σ deg(v) = 2|E|, |E| = 6

    def test_grado(self):
        self.assertAlmostEqual(redes.grado(R, "D"), 3 / 5)

    def test_cercania(self):
        # R es conexo: la armónica de D es (1/1·3 + 1/2·2)/5 = 4/5
        self.assertAlmostEqual(redes.cercania(R, "D"), (1 + 1 + 1 + 0.5 + 0.5) / 5)

    def test_intermediacion(self):
        cb = redes.intermediacion(R)
        self.assertAlmostEqual(cb["E"], 4)    # F con A, B, C, D: todos pasan por E
        self.assertAlmostEqual(cb["D"], 6.5)  # 6 pares {A,B,C}x{E,F} + la mitad del par {B, C}
        self.assertAlmostEqual(cb["A"], 0.5)  # la otra mitad de {B, C}
        self.assertAlmostEqual(cb["F"], 0)
        self.assertAlmostEqual(redes.intermediacion(R, normalizada=True)["E"], 4 / 10)


class TestSustitucion(unittest.TestCase):
    def test_mismo_modelo_en_otro_proveedor_es_vecino(self):
        adj = redes.construir([("gemini/gemma-4-31b-it", "gemma-4-31b-it", 0.84),
                               ("nvidia/google/gemma-4-31b-it", "google/gemma-4-31b-it", 0.84),
                               ("ovh/llama", "llama-3.3", 0.50)])
        self.assertIn("nvidia/google/gemma-4-31b-it", adj["gemini/gemma-4-31b-it"])
        self.assertEqual(adj["ovh/llama"], set())  # componente aislada: nadie lo sustituye

    def test_orden_respaldo_por_capas(self):
        adj = {"e": {"a"}, "a": {"e", "b"}, "b": {"a"}, "z": set()}
        orden = redes.orden_respaldo(adj, "e", {"a": 2, "b": 0, "z": 0})
        self.assertEqual(orden, ["a", "b", "z"])  # capa 1, capa 2, luego no alcanzable


if __name__ == "__main__":
    unittest.main()
