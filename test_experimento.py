"""El calificador del experimento: si se equivoca, los resultados no valen nada."""
import unittest

from experimento import es_correcta, respuesta_final


class TestCalificador(unittest.TestCase):
    def test_extrae_respuesta_final(self):
        self.assertEqual(respuesta_final("Por el lema...\n**RESPUESTA FINAL: 30**"), "30")
        self.assertEqual(respuesta_final("bla\nRespuesta final: Sí."), "si")

    def test_numeros(self):
        self.assertTrue(es_correcta("30 aristas", "30"))
        self.assertTrue(es_correcta("3/5", "0.6"))
        self.assertTrue(es_correcta("0,6", "0.6"))
        self.assertFalse(es_correcta("31", "30"))
        self.assertFalse(es_correcta("no se", "30"))

    def test_palabras(self):
        self.assertTrue(es_correcta("si", "sí"))
        self.assertTrue(es_correcta("aciclico", "acíclico"))
        self.assertTrue(es_correcta("bfs (busqueda en anchura)", "BFS"))
        self.assertFalse(es_correcta("no", "sí"))
        self.assertFalse(es_correcta("nodo", "no"))  # palabra completa, no prefijo


if __name__ == "__main__":
    unittest.main()
