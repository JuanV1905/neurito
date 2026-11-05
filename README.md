# Neurito: mi cerebro artificial

Neurito es una mascota controlada por redes neuronales hechas solo con NumPy. El proyecto tiene tres partes: la mascota original, que decide si comer, dormir o jugar según sus necesidades; Neurito 2.0, que aprende cuándo saltar obstáculos; y Neurito 3.0, que aprende a navegar un laberinto para buscar comida. La interfaz gráfica de las tres está hecha con Pygame.

## Partes del proyecto

- **Mascota original** (`main.py`): decide entre comer, dormir y jugar.
- **Neurito 2.0** (`saltar.py`): aprende a saltar obstáculos. Está explicado en `GUIA_SALTO.md`.
- **Neurito 3.0** (`laberinto.py`): aprende a buscar comida en un laberinto esquivando paredes. Está explicado en `GUIA_LABERINTO.md`.

## Cómo ejecutarlo

**Mascota original:**
```bash
uv run main.py
```

**Neurito 2.0 (Salto):**
```bash
uv run saltar.py
```

**Neurito 3.0 (Laberinto):**
```bash
uv run laberinto.py
```

Al empezar cada programa se entrena la red (se ve la pérdida en la terminal) y luego abre la ventana del juego. En `main.py`, se puede cambiar el idioma de la interfaz cambiando `LANGUAGE` a `"es"`.

## Cómo funciona la Mascota Original

- **Entrada:** 3 necesidades entre 0 y 1.
- **Capa oculta:** 6 neuronas con ReLU.
- **Salida:** 3 neuronas con Softmax (comer, dormir, jugar).
- **Datos:** 3000 ejemplos sintéticos. La acción correcta es la necesidad más grande, usando `cansancio = 1 - energía` para que las tres vayan en el mismo sentido.
- **Entrenamiento:** descenso de gradiente con todos los datos, 1200 épocas y tasa de aprendizaje 0.1.

## Parámetros que se pueden cambiar

Están al inicio de `main.py`: `HIDDEN_NEURONS`, `EPOCHS`, `LEARNING_RATE`, `NEUTRAL_LIMIT` y `UNSURE_LIMIT`. En los experimentos vimos que con 3 neuronas la red es inestable, con 6 alcanza, y que una tasa de aprendizaje de 1 o más hace el entrenamiento inestable.

## Qué se agregó después de los experimentos

- Parámetros de entrenamiento como constantes con comentarios.
- La política usa el cansancio de forma explícita en una sola función, que sirve para entrenar y para probar.
- Además de la precisión, ahora se imprime la pérdida final, la línea base, la precisión en casi empates y la matriz de confusión, porque una precisión alta sola puede engañar.
- En pantalla se muestra la confianza de la red y avisa cuando está dudando.

## Neurito 2.0: aprende a saltar

En esta parte Neurito ve un obstáculo que se acerca y tiene que decidir cuándo saltar.

- **Entrada:** 3 datos normalizados: distancia al obstáculo, velocidad y tiempo estimado hasta llegar (distancia / velocidad).
- **Red:** arquitectura 3 → 12 → 1, con 12 neuronas ocultas con ReLU y una salida con sigmoide. Si la salida es mayor a 0.5, salta.
- **Aprendizaje:** supervisado, con datos sintéticos. La etiqueta dice que hay que saltar cuando el tiempo estimado está entre 7 y 13 fotogramas. No es aprendizaje por refuerzo.
- **Entrenamiento:** entropía cruzada binaria, retropropagación y descenso de gradiente, con las clases balanceadas para que la red no aprenda a decir siempre "esperar".
- **Controles:** `1` entrena una época, `2` cien épocas, `3` entrenamiento continuo, `A` cambia entre automático y manual, `Espacio` salta, `C` reinicia la pista, `R` reinicia los pesos y `Esc` sale.

Algo importante es que una precisión alta no garantiza que el juego salga bien, porque el resultado también depende de la física del salto. Esa es la pregunta central de la guía.

Archivos de esta sección: `saltar.py` (juego y visualización), `salto_modelo.py` (red y datos), `salto_juego.py` (física y colisiones) y `GUIA_SALTO.md`.

## Neurito 3.0: aprende a buscar comida en el laberinto

En esta sección Neurito utiliza sensores de distancia para detectar paredes y encontrar la ruta hacia la comida sin chocar.

- **Entrada:** sensores de proximidad (frente, izquierda, derecha) y distancia/orientación hacia la comida, normalizados entre 0 y 1.
- **Red:** arquitectura multicapa con activación ReLU en la capa oculta y Softmax en la salida.
- **Salida:** probabilidades para las acciones de movimiento (Avanzar, Girar Izquierda, Girar Derecha, Reposo).
- **Aprendizaje:** supervisado con reglas de navegación libre y evasión de obstáculos.
- **Controles:** `A` activa/desactiva el modo automático, `R` reinicia la posición o los pesos de la red, `1`-`3` para ajustar iteraciones de entrenamiento.

Archivos de esta sección: `laberinto.py`, `laberinto_red.py`, `laberinto_mundo.py`, `test_laberinto.py` y `GUIA_LABERINTO.md`.

## Reportes de experimentos

Los experimentos detallados y sus respuestas teóricas se dividen según cada módulo:
- Experimento Mascota: `EXPERIMENTO_NEURITO1.md`
- Experimento Salto: `EXPERIMENTO_NEURITO2.md`
- Experimento Laberinto: `EXPERIMENTO_NEURITO3.md`