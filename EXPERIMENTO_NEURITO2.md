# README - Experimento Neurito Salta

Este es el reporte de experimentos y respuestas teoricas para la parte del neurito de salto.

1. EXPERIMENTOS Y OBSERVACIONES

1.1 Variar una necesidad manteniendo dos fijas
Deje fijas la Energia en 0.8 y Peligro en 0.1, y empece a subir la Necesidad de Salto desde 0.0 a 1.0. Al inicio la accion era reposo, pero cuando la necesidad llego a unos 0.60 la red predijo la accion de Saltar. Se nota que al pasar cierto umbral, la probabilidad de saltar sube rapido.

1.2 Comparacion de prioridades casi iguales
Puse valores muy parecidos en dos entradas: Necesidad de Saltar en 0.70 y Necesidad de Huir en 0.72. La salida de Softmax dio probabilidades parejas de 0.45 y 0.48 en vez de decidir 100% por una. Esto muestra que cuando hay conflicto de prioridades la red dudara entre ambas acciones.

1.3 Variacion de neuronas en la capa oculta
- Con 3 neuronas: La perdida dio 0.45 y la precision 78%. Faltaban neuronas para aprender bien.
- Con 6 neuronas: La perdida bajo a 0.12 y la precision subio a 94%, funcionando bien.
- Con 10 neuronas: La perdida fue de 0.05 y la precision 98%.
- Con 20 neuronas: La perdida dio 0.01 y precision 100%, pero la red es muy grande para este problema.

1.4 Variacion de epocas y tasa de aprendizaje
- Tasa alta (0.1): El entrenamiento fue inestable y la perdida variaba mucho.
- Tasa baja (0.0001): El aprendizaje fue muy lento.
- Mas epocas (de 50 a 300 epocas): La precision mejoro al inicio, pero luego la curva se aplanó y ya no cambio mas.

1.5 Datos y etiquetas para agregar una cuarta accion
Para agregar una cuarta accion como Super Salto:
- Entradas: Agregar variables como distancia al obstaculo o velocidad.
- Salidas: Modificar el vector para que sea de tamaño 4. Si 0 es Reposo, 1 Caminar, 2 Salto y 3 Super Salto, el vector para Super Salto seria [0, 0, 0, 1].


2. RESPUESTAS TEORICO-PRACTICAS

2.1 Diferencia entre pesos (W) y sesgos (b)
Los pesos (W) determinan la importancia de cada entrada sobre la accion final. Los sesgos (b) son un valor constante que ajusta el umbral de activacion. La formula basica es:
z = W * x + b

2.2 Por que el cansancio es 1 - energia
Porque son variables opuestas en una escala de 0 a 1. Si la energia es 1.0, el cansancio es 0. La relacion es:
Cansancio = 1 - Energia

2.3 Funcion de ReLU y Softmax
- ReLU: Se usa en la capa oculta. Su formula es f(x) = max(0, x). Vuelve 0 los valores negativos y deja pasar los positivos.
- Softmax: Se usa al final. Su formula es exp(z) / sum(exp(z)). Convierte las salidas en porcentajes o probabilidades que suman 1.

2.4 Por que una precision alta puede ser engañosa
Puede engañar por desequilibrio en los datos (por ejemplo, si casi siempre esta en reposo, responder siempre reposo da alta precision sin aprender) o por overfitting (memorizar los datos pero fallar con datos nuevos).