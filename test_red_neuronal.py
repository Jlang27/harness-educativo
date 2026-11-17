"""Casos de resultado conocido para la red neuronal. Ejecutar: python -m unittest -v"""
import json
import time
import unittest

import red_neuronal


class TestRedNeuronal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inicio = time.time()
        cls.red = red_neuronal.red()
        cls.segundos = time.time() - inicio

    def test_aprende_el_conjunto_de_entrenamiento(self):
        ejemplos = json.loads(red_neuronal.DATOS.read_text(encoding="utf-8"))["ejemplos"]
        aciertos = sum((self.red.probabilidad(e["texto"]) >= 0.5) == bool(e["calculo"]) for e in ejemplos)
        self.assertGreaterEqual(aciertos / len(ejemplos), 0.95)

    def test_frases_nuevas(self):
        # Ninguna de estas frases está en entrenamiento.json.
        self.assertGreater(self.red.probabilidad("Demuestra por inducción que 2^n > n para todo n natural"), 0.5)
        self.assertGreater(self.red.probabilidad("calcula la derivada de e^(2x)"), 0.5)
        self.assertLess(self.red.probabilidad("Hazme un resumen de la historia de Roma"), 0.5)
        self.assertLess(self.red.probabilidad("revisa la redacción de mi ensayo de literatura"), 0.5)

    def test_reproducible_y_rapida(self):
        ejemplo = [("calcula 2+2", 1), ("resume el libro", 0)]
        a, b = red_neuronal.Red(ejemplo), red_neuronal.Red(ejemplo)
        self.assertEqual(a.probabilidad("calcula 3+3"), b.probabilidad("calcula 3+3"))  # misma semilla, mismos pesos
        self.assertLess(self.segundos, 10)


if __name__ == "__main__":
    unittest.main()
