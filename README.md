# Harness educativo

Elige el modelo de IA gratuito adecuado para la tarea de un estudiante, explica por qué lo eligió y abre un chat con él. Funciona desde el primer arranque sin crear cuentas. Tiene una interfaz gráfica estilo app de chat: barra lateral con los chats, respuestas con Markdown y fórmulas, modo claro y oscuro.

## Ejecutar (usuario final)
Descarga el repositorio con **Code → Download ZIP**, extrae la carpeta completa y abre `dist/HarnessEducativo.exe` con doble clic. También puedes descargar únicamente ese ejecutable desde su página en GitHub. Se abre una ventana con la app. Para salir, cierra la ventana; el programa se apaga solo unos dos minutos después.

1. **Funciona sin configurar nada:** trae incluido OVHcloud AI Endpoints, que no pide cuenta ni clave. Su cupo es bajo: unos 2 mensajes por minuto por modelo.
2. Elige la modalidad (Estudiar, Practicar o Revisar), adjunta tu material con el clip si quieres y escribe la pregunta inicial. Al crear el chat, la red neuronal estima si la tarea es de cálculo; el interruptor "Necesito máxima precisión" lo exige desde el inicio. Si después cambias a una tarea de otro tipo, inicia un chat nuevo para que se vuelva a seleccionar el modelo.
3. En el selector de la barra de mensaje, **Encontrar modelo** deja que el harness elija. También puedes elegir un modelo concreto.
4. Al empezar el chat, una tarjeta explica por qué se eligió ese modelo, con sus datos reales. Si un proveedor sin configurar tiene un modelo mejor, la tarjeta recomienda conseguir esa clave gratuita y trae el enlace.
5. Para mejores modelos y más cupo, abre **Configurar proveedores** y pega la clave gratuita de Google Gemini, NVIDIA NIM, Groq, Cerebras, Mistral u OpenRouter, o la de DeepSeek (de pago). También puedes **agregar un proveedor personalizado**: cualquier API compatible con OpenAI, con su nombre, URL base y clave. Los modelos de cada proveedor se descubren solos; los que no están calificados en `modelos.json` aparecen como "sin calificar" y el harness no los usa para cálculo. Si un modelo se queda sin cuota, la app pasa al siguiente recomendado, aunque sea de otro proveedor, y lo avisa en el chat.

Los chats y las claves se guardan en la carpeta `datos/`, junto al `.exe`. Si pones un `modelos.json` junto al `.exe`, se usa ese en lugar del que viene incluido.

Requisitos: Windows 10 u 11 con Microsoft Edge, que viene instalado de fábrica. Si no hay Edge, se abre el navegador por defecto. Necesita internet para hablar con los modelos.

## Desarrollo
Requiere Python 3.10 o superior. La app solo usa la librería estándar; la red neuronal también está hecha a mano.
```bash
python app.py
```
Pruebas con resultado conocido: las reglas de decisión, la red neuronal (con frases que no vio en el entrenamiento) y el cambio de modelo cuando uno agota la cuota (se prueba contra una API falsa local).
```bash
python -m unittest -v
```
Para volver a generar el `.exe`, primero instala PyInstaller con `pip install pyinstaller` y luego:
```bash
python -m PyInstaller --noconfirm --onefile --noconsole --name HarnessEducativo --add-data "index.html;." --add-data "modelos.json;." --add-data "entrenamiento.json;." app.py
```

## Experimento (resultados para el documento)
Compara tres estrategias sobre 20 ejercicios con respuesta conocida, tomados de los apuntes del curso (`experimento.json`): **siempre el modelo más potente**, **un modelo al azar** y **el harness**. Registra aciertos, cortes y tokens (reportados por la API o estimados si no informa uso), y calcula un costo de referencia. Este costo no representa un cobro efectivo al usar cupos gratuitos.
```bash
python experimento.py
```
Opciones: `--n 5` usa solo 5 ejercicios; `--pausa 6` espera 6 s entre llamadas, para no agotar los cupos gratuitos. Los resultados quedan en `resultados_experimento.csv` y `.json`. Usa las claves guardadas por la app (`datos/config.json` o `dist/datos/config.json`).

Los resultados incluidos se obtuvieron antes de incorporar el respaldo por BFS, cuando se ordenaba solo por costo. Describen una ejecución de 20 ejercicios y no evalúan específicamente el orden de respaldo actual. No registran las causas exactas de los cortes ni todos los intentos fallidos. Ejecutar de nuevo el experimento utiliza los proveedores configurados y reemplaza los archivos de resultados.

## Cómo decide (teoría del curso)
La clasificación y la selección por costo se realizan una sola vez, al iniciar cada chat con **Encontrar modelo**. Después se comprueba el contexto y se avanza por la cadena de respaldo si hace falta. GPQA y los umbrales son indicadores aproximados del catálogo; no garantizan respuestas correctas.

1. **Red neuronal** (`red_neuronal.py`): un perceptrón multicapa (entrada dispersa → 8 neuronas tanh → salida sigmoide) entrenado con retropropagación sobre `entrenamiento.json`. Estima N = P(la pregunta exige cálculo o demostraciones).
2. **Requisitos mínimos:** la entrada inicial exige precisión si la red neuronal lo detecta o si el estudiante activa el interruptor. Un modelo es apto si está disponible, su contexto alcanza y su puntaje de razonamiento es ≥ 0.80 cuando se exige precisión, o ≥ 0.60 si no. Se usa un margen de contexto de 1.5 al seleccionar y de 1.2 al revisar el historial en cada envío; el cupo real se comprueba al pedir la respuesta.
3. **Grafo bipartito ponderado:** tipos de tarea ↔ modelos. Cada arista lleva el **costo** estimado de atender la tarea con ese modelo (tokens × precio de referencia) y su **calidad** w(t, m). Gana la arista apta de **menor costo** (si hay empate, la de mayor w), y el resto queda como cadena de respaldo.
4. **Ventana deslizante:** sin ventana, el mensaje j reenvía los j anteriores, y el total tras n mensajes es t·n(n+1)/2 (cuadrático, demostrable por inducción). Con una ventana de k mensajes, el costo crece de forma lineal.
5. **Recomendación:** se repite la elección suponiendo que los proveedores sin clave sí estuvieran conectados. Si gana uno de ellos con más peso, se sugiere conseguir esa clave.
6. **En ejecución:** si el modelo elegido responde 401, 402, 403, 404, 413, 429 o 5xx (clave inválida, cuota agotada o caído), la app pasa al siguiente de la cadena, aunque sea de otro proveedor, y lo avisa en el chat.

Todos los proveedores usan APIs compatibles con OpenAI, así que un mismo cliente (`cliente.py`) sirve para todos. La idea es la misma que la de FreeLLMAPI, pero implementada aquí como programa propio.

## Archivos
- `router.py`: los requisitos mínimos y el grafo bipartito ponderado.
- `red_neuronal.py` y `entrenamiento.json`: la red neuronal y sus ejemplos etiquetados.
- `redes.py`: la red de sustitución entre modelos (Unidad 4): BFS, grado, cercanía e intermediación, y el orden de respaldo.
- `documento/`: el informe en LaTeX (`informe.tex`, `referencias.bib`) y su PDF compilado.
- `cliente.py`: las llamadas HTTP a la API y el cruce del catálogo con los modelos que reporta cada proveedor.
- `app.py`: los proveedores, la recomendación, el servidor local (solo escucha en 127.0.0.1) y la apertura de la ventana.
- `index.html`: toda la interfaz gráfica.
- `modelos.json`: los modelos calificados. `razonamiento` = GPQA Diamond / 100, con la fuente de cada puntaje en el campo `fuente` ("estimado" = sin dato publicado, hay que verificarlo).
- `experimento.py` y `experimento.json`: el experimento antes/después y sus ejercicios.
- `test_router.py`, `test_red_neuronal.py`, `test_redes.py` (verifica el grafo R de la Unidad 4), `test_app.py` y `test_experimento.py`: las pruebas.
