# README - Experimento Neurito Laberinto (Busqueda de Comida)

Este documento es el reporte de experimentos y respuestas teoricas para la seccion donde el Neurito aprende a navegar en un laberinto para encontrar comida.

1. EXPERIMENTOS Y OBSERVACIONES

1.1 Variar sensores manteniendo constante la salud y hambre
Se mantuvieron la Salud en 1.0 y el Hambre en 0.8. Se variaron los sensores de pared (izq, frente, der) y la distancia a la comida. Cuando los sensores no detectaban paredes enfrente y la distancia a la comida disminuia, la red predecia la accion de Avanzar con alta probabilidad. Sin embargo, al poner una pared directamente enfrente (sensor frente = 1.0), la probabilidad de Avanzar cayo a cerca de 0 y subio la probabilidad de Girar (Izquierda o Derecha según cuál espacio estuviera libre).

1.2 Dilema entre Peligro/Obstaculo y Hambre alta
Se configuro Hambre en 0.95 (necesidad critica de comida) pero con una pared u obstaculo muy cercano al frente. La salida Softmax dio valores divididos (por ejemplo, Avanzar 0.42 y Girar 0.48). Esto evidencia un conflicto de decision: la necesidad de buscar comida empuja a avanzar, pero la senal del sensor frena el movimiento para evitar una colision.

1.3 Variacion de neuronas en la capa oculta del laberinto

* Con 4 neuronas: La perdida fue de 0.52 y la precision de 70%. La red fallaba seguido en esquinas complejas.

* Con 8 neuronas: La perdida bajo a 0.15 y la precision subio a 92%, resolviendo bien la navegacion basica.

* Con 16 neuronas: La perdida cayo a 0.03 y la precision alcanzo el 98%.

* Con 32 neuronas: La perdida fue de 0.01 y precision 100%, aunque representa una sobrecapacidad para el mapa simple del laberinto.

1.4 Variacion de parametros de entrenamiento (Epocas y Learning Rate)

* Tasa alta (0.1): El agente oscilaba entre acciones y el entrenamiento no convergia bien.

* Tasa muy baja (0.0001): Requirio muchas iteraciones para empezar a girar adecuadamente ante paredes.

* Epocas (de 50 a 400): Con 50 epocas el agente chocaba en callejones sin salida. A partir de las 200 epocas, la tasa de exito para encontrar comida se estabilizo.

1.5 Adaptacion de entradas y salidas para nuevas acciones
Para permitir que el Neurito coma directamente o use un radar de comida:

* Entradas: Se agregan sensores de direccion de la comida (angulo) y sensor de presencia de comida inmediata.

* Salidas: Se amplia el vector a 5 acciones: [0: Reposo, 1: Avanzar, 2: Girar Izq, 3: Girar Der, 4: Comer]. La etiqueta para la accion Comer seria [0, 0, 0, 0, 1].

2. RESPUESTAS TEORICO-PRACTICAS

2.1 Funcion de los pesos (W) y sesgos (b) en la navegacion
Los pesos (W) ponderan la importancia de cada sensor (paredes, distancia a comida, hambre) para decidir la direccion. Los sesgos (b) permiten fijar el umbral minimo de estimulo necesario para que el agente cambie de direccion o empiece a mover. La ecuacion es:
z = W * x + b

2.2 Normalizacion de distancias y sensores
Las distancias del laberinto y los sensores de proximidad se escalan entre 0.0 y 1.0. Esto evita que una distancia grande en pixeles o casillas domine indebidamente sobre variables como el hambre o la salud.

2.3 Funciones de Activacion (ReLU y Softmax)

* ReLU: Aplicada en la capa oculta, f(x) = max(0, x). Permite activar combinaciones especificas de sensores (por ejemplo, activar una ruta solo si "frente libre" Y "comida cerca").

* Softmax: Aplicada en la salida. Convierte el resultado en una distribucion de probabilidad sobre las acciones de movimiento (Avanzar, Girar Izq, Girar Der, Reposo), sumando 1.0 en total.

2.4 Por que la precision no garantiza llegar siempre a la comida
Una precision alta en el conjunto de entrenamiento indica que el modelo clasifica bien las decisiones locales (evitar choque puntual). Sin embargo, esto no garantiza evitar bucles infinitos en el laberinto ni encontrar la ruta optima a la comida si el agente cae en situaciones no contempladas durante el entrenamiento.