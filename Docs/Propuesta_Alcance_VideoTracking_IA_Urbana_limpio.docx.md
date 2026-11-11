

**PROPUESTA DE PROYECTO — DEFINICIÓN DE ALCANCE**

**Video tracking e IA urbana:**

**Seguimiento multiobjeto y generación automática de reportes para seguridad y movilidad en la ciudad**

# **1\. Presentación y contexto**

Las ciudades intermedias colombianas disponen cada vez de más cámaras en vías, parques y accesos vehiculares, pero la información que producen se revisa casi siempre de forma manual y reactiva: alguien mira el video después de que ocurrió algo. Esto desaprovecha el video como fuente de datos para dos problemas cotidianos de la administración municipal: la seguridad (detectar situaciones anómalas) y la movilidad (saber cuántos y qué tipo de actores circulan por un punto, en qué sentido y a qué horas).

Este proyecto propone construir un prototipo funcional de bajo costo que, a partir de video de una escena urbana, detecte y clasifique los actores viales presentes, los siga cuadro a cuadro asignándoles un identificador único (seguimiento multiobjeto, MOT) y genere de forma automática un reporte con conteos, flujos y alertas básicas. El énfasis del trabajo está en la evaluación cuantitativa del sistema: qué tan bien clasifica, dónde se equivoca y por qué.

# **2\. Objetivos**

## **2.1 Objetivo general**

Desarrollar y evaluar un prototipo de seguimiento multiobjeto con clasificación multiclase sobre video urbano, capaz de generar automáticamente reportes de seguridad y movilidad, midiendo su desempeño mediante métricas estándar de detección, clasificación y tracking.

## **2.2 Objetivos específicos**

* Construir un conjunto de datos propio de 50 a 100 registros de video (clips cortos) capturados en campo en puntos representativos de la ciudad, debidamente anotados.

* Definir un esquema de clases multiclase acorde al tamaño del conjunto de datos y al contexto vial colombiano.

* Implementar y comparar al menos tres arquitecturas de redes neuronales para la tarea de detección/clasificación, usando transferencia de aprendizaje.

* Integrar un algoritmo de seguimiento multiobjeto que mantenga la identidad de cada actor a lo largo del clip.

* Evaluar el sistema con métricas por clase y globales, y construir la matriz de confusión para identificar los modos de falla.

* Generar automáticamente un reporte estructurado (conteos por clase, sentido de circulación, franja horaria y alertas) exportable a CSV/PDF.

# **3\. Alcance del proyecto**

Dado que se trata de un proyecto académico semestral, con recursos computacionales limitados y un conjunto de datos pequeño, el alcance se delimita explícitamente para que los resultados sean verificables y honestos.

## **3.1 Lo que el proyecto sí incluye**

* Captura propia en campo de 50–100 clips de video (10–20 s cada uno) en 2 o 3 puntos fijos de la ciudad, en distintas franjas horarias.

* Anotación manual de cuadros clave con cajas delimitadoras (bounding boxes) y etiqueta de clase.

* Entrenamiento por transferencia (fine-tuning) de modelos preentrenados en COCO; no se entrena desde cero.

* Seguimiento multiobjeto sobre el clip, con asignación y mantenimiento de ID.

* Conteo por línea virtual (line crossing) y por zona de interés (ROI).

* Reporte automático: conteo total y por clase, distribución por franja horaria, sentido de circulación y alertas de reglas simples.

* Evaluación cuantitativa completa: métricas por clase, métricas globales, matriz de confusión y análisis de errores.

* Anonimización de rostros y placas en los datos, conforme a la Ley 1581 de 2012\.

## **3.2 Lo que el proyecto no incluye**

* Reconocimiento facial, identificación de personas o lectura de placas (OCR). El sistema trabaja a nivel de categoría, no de identidad.

* Despliegue en producción, integración con la red de cámaras del municipio ni operación 24/7.

* Reidentificación entre cámaras distintas (cross-camera Re-ID).

* Predicción de accidentes, detección de conductas delictivas específicas o inferencia de intenciones.

* Entrenamiento de arquitecturas nuevas desde cero ni contribuciones al estado del arte.

* Estimación de velocidad con precisión metrológica; solo se reporta una estimación relativa y no calibrada.

## **3.3 Supuestos y restricciones**

* Cámara fija, altura de 3 a 6 m, ángulo cenital-oblicuo, sin movimiento de paneo durante la captura.

* Condiciones diurnas y de baja/media lluvia; no se cubre visión nocturna ni infrarroja.

* Escenas de densidad baja a media (idealmente menos de 15 objetos simultáneos por cuadro).

* Entrenamiento en GPU gratuita (Google Colab) con tiempos de sesión limitados, lo que restringe el tamaño de los modelos y el número de épocas.

# **4\. Conjunto de datos**

El conjunto de datos está compuesto por 50–100 registros, entendiendo por “registro” un clip de video corto anotado. De cada clip se extraen cuadros muestreados (por ejemplo, 1 de cada 10), lo que produce un volumen de instancias etiquetadas suficiente para un ajuste fino sin ser un corpus de escala industrial.

| Aspecto | Definición | Justificación |
| ----- | :---: | :---: |
| Unidad de registro | 1 clip de 10–20 s a 25–30 fps | Permite evaluar tracking, no solo detección |
| Cantidad | 50–100 clips | Alcance realista para captura y anotación manual |
| Cuadros anotados | \~15–25 por clip (muestreo 1:10) | Evita cuadros casi idénticos y redundantes |
| Instancias estimadas | \~1.000–1.500 cajas etiquetadas | Suficiente para fine-tuning con congelamiento parcial |
| Partición | 70 % entrenamiento / 15 % validación / 15 % prueba | Partición a nivel de clip, nunca de cuadro |
| Formato | YOLO (.txt) y COCO (.json) | Compatibilidad con los tres modelos evaluados |
| Herramienta | CVAT o Roboflow (versión gratuita) | Anotación asistida y exportación multiformato |

**Nota metodológica clave:** la partición debe hacerse por clip completo y no por cuadro. Si cuadros del mismo clip quedan repartidos entre entrenamiento y prueba, el modelo “ya vio” la escena y las métricas quedan infladas (fuga de datos). Con un conjunto tan pequeño, además, se recomienda validación cruzada estratificada de k \= 5 en lugar de una sola partición, y reportar media ± desviación estándar.

# **5\. Definición de clases**

Con 50–100 registros no es viable sostener un esquema de 15 o 20 clases: las clases poco frecuentes quedarían con una decena de ejemplos y sus métricas no serían interpretables. Se propone por tanto un esquema de seis clases mutuamente excluyentes, agrupadas por función vial, con la posibilidad de colapsar a cuatro si alguna clase no alcanza un mínimo de ejemplos.

| \# | Clase | Descripción operativa | Uso en el reporte |
| ----- | :---: | :---: | :---: |
| 1 | Peatón | Persona caminando, corriendo o detenida en vía o andén | Aforo peatonal, seguridad |
| 2 | Bicicleta | Ciclista con su vehículo (unidad ciclista+bici) | Movilidad activa, ciclorruta |
| 3 | Motocicleta | Motociclista con su vehículo, incluye parrillero | Alta relevancia en siniestralidad |
| 4 | Automóvil | Vehículo liviano: carro, taxi, camioneta, SUV | Aforo vehicular, congestión |
| 5 | Bus | Transporte público colectivo: bus, buseta, microbús | Frecuencia de transporte público |
| 6 | Camión | Vehículo de carga: camión, volqueta, furgón | Restricción de carga pesada |

**Plan de contingencia:** si en el conjunto de prueba alguna clase queda con menos de 15 instancias, se colapsa el esquema a cuatro clases —Peatón, Vehículo de dos ruedas (bicicleta \+ motocicleta), Vehículo liviano y Vehículo pesado (bus \+ camión)— y se documenta la decisión. Es preferible reportar cuatro clases bien medidas que seis mal medidas.

Se maneja además una categoría implícita de fondo (background): toda región que el modelo no asigna a ninguna de las seis clases. Aparece en la matriz de confusión como columna “No detectado” y captura los falsos negativos.

# **6\. Técnicas para la clasificación multiclase**

La clasificación es multiclase de etiqueta única: cada objeto detectado pertenece a exactamente una de las seis clases. Las técnicas seleccionadas responden a la restricción dominante del proyecto, que es la escasez de datos.

## **6.1 Transferencia de aprendizaje (fine-tuning)**

Es la técnica central y no negociable. Se parte de pesos preentrenados en COCO —conjunto que ya contiene person, bicycle, motorcycle, car, bus y truck, es decir, exactamente el esquema propuesto— y se ajustan únicamente las últimas capas y la cabeza de clasificación, congelando el extractor de características durante las primeras épocas. Esto permite converger con cientos de ejemplos en lugar de decenas de miles.

## **6.2 Función de pérdida y salida softmax**

La cabeza de clasificación termina en una capa softmax de seis salidas con entropía cruzada categórica, que fuerza que las probabilidades sumen 1 y modela correctamente la exclusividad entre clases. Como el conjunto estará desbalanceado por naturaleza (habrá muchos más automóviles que camiones), se aplica ponderación de clases inversamente proporcional a la frecuencia, o bien focal loss, que reduce el peso de los ejemplos fáciles y obliga al modelo a atender los casos minoritarios.

## **6.3 Aumento de datos (data augmentation)**

Con 50–100 registros, el aumento multiplica efectivamente el conjunto. Se aplican transformaciones geométricas (volteo horizontal, escalado, recorte, rotación leve), fotométricas (variación de brillo, contraste, matiz y saturación para simular distintas horas del día) y compuestas (mosaic y mixup, nativas en el pipeline de YOLO). No se aplica volteo vertical, porque invierte la gravedad de la escena y genera ejemplos irreales.

## **6.4 Clasificador clásico sobre características profundas**

Alternativa útil precisamente cuando hay pocos datos: se congela por completo la red convolucional, se extrae el vector de características (embedding) de cada recorte y sobre él se entrena un clasificador clásico —SVM con kernel RBF en esquema uno-contra-todos (One-vs-Rest), o Random Forest—. Al tener muchos menos parámetros que una cabeza densa, tiende a sobreajustar menos y sirve como línea base honesta contra la cual comparar el fine-tuning completo.

## **6.5 Clasificación en cascada**

Para las parejas de clases que más se confunden (bus/camión y bicicleta/motocicleta), se propone una segunda etapa: el detector resuelve la categoría gruesa y un clasificador ligero especializado, entrenado solo con los recortes de esas dos clases, decide la etiqueta fina. Es una mejora de bajo costo que suele levantar varios puntos el F1 de las clases minoritarias.

## **6.6 Validación**

Validación cruzada estratificada de k \= 5 a nivel de clip, reportando media y desviación estándar de cada métrica. Con conjuntos pequeños, un único resultado puede variar varios puntos solo por el azar de la partición; reportar la dispersión es parte de la honestidad del experimento.

# **7\. Arquitecturas de redes neuronales propuestas**

Se seleccionan tres arquitecturas con filosofías de diseño distintas, de modo que la comparación sea informativa y no una simple variación de hiperparámetros. Todas se usan con pesos preentrenados.

## **7.1 YOLOv8n / YOLO11n — CNN de una etapa (modelo principal)**

Red convolucional anchor-free de una sola etapa: predice cajas y clases en una única pasada. Es el modelo de referencia del proyecto por su balance entre velocidad y precisión. La variante nano (\~3,2 M de parámetros) entrena en minutos en Colab y alcanza tiempo real (30+ fps) incluso en CPU modesta. Su punto débil son los objetos pequeños y las oclusiones fuertes.

**Rol:** detección y clasificación de las seis clases, en línea, sobre todo el flujo de video.

## **7.2 Faster R-CNN (ResNet-50 \+ FPN) — CNN de dos etapas (línea base de precisión)**

Arquitectura de dos etapas: una red de propuestas de región (RPN) genera candidatos y una segunda cabeza los clasifica y refina. Es sensiblemente más lenta (5–10 fps con GPU) pero históricamente más precisa en objetos pequeños y escenas densas, gracias a la pirámide de características (FPN). En este proyecto no busca ser el modelo desplegado, sino el techo de precisión contra el cual medir cuánto se pierde al optar por YOLO.

**Rol:** referencia de exactitud; especialmente útil para peatones y bicicletas lejanos a la cámara.

## **7.3 MobileNetV3-Small / EfficientNet-B0 — CNN clasificadora ligera (cascada)**

No es un detector sino un clasificador de imagen: recibe el recorte que produjo el detector y emite la clase. Con menos de 3 M de parámetros y bloques separables en profundidad, entrena con muy pocos ejemplos y es el candidato natural para la segunda etapa de la cascada descrita en 6.5, resolviendo las confusiones bus/camión y bicicleta/motocicleta.

**Rol:** clasificador especializado de segunda etapa sobre recortes; también línea base pura de clasificación multiclase.

## **7.4 Alternativa opcional: RT-DETR (Transformer)**

Si el cronograma lo permite, se puede añadir RT-DETR, detector basado en Transformer que elimina la supresión de no-máximos (NMS) y maneja mejor las oclusiones mediante atención global. Se considera opcional porque los Transformers son notoriamente ávidos de datos y con 50–100 registros es probable que rinda por debajo de YOLO, lo cual sería en sí mismo un resultado interesante para discutir.

## **7.5 Seguimiento multiobjeto**

El tracking se resuelve con el paradigma tracking-by-detection: el detector entrega las cajas de cada cuadro y el rastreador las asocia en el tiempo.

* ByteTrack — asocia por IoU y filtro de Kalman, aprovechando también las detecciones de baja confianza. No requiere red neuronal adicional ni entrenamiento. Es la opción recomendada por su simplicidad.

* DeepSORT con OSNet — añade una red de reidentificación que compara la apariencia visual. Mejora el mantenimiento de ID tras oclusiones prolongadas, a costa de velocidad y de necesitar el módulo Re-ID preentrenado.

# **8\. Correspondencia entre modelo, clases y métricas**

La siguiente tabla resume qué modelo atiende qué clases y con qué métricas se evalúa cada uno. Es el núcleo del diseño experimental.

| Modelo | Tipo de red | Clases / rol asignado | Métricas principales |
| ----- | :---: | :---: | :---: |
| YOLOv8n / YOLO11n | CNN una etapa, anchor-free | Las 6 clases; detector principal en línea | mAP@0.5, mAP@0.5:0.95, F1 macro, fps |
| Faster R-CNN (ResNet-50+FPN) | CNN dos etapas con RPN | Las 6 clases; línea base de precisión, énfasis en objetos pequeños (peatón, bicicleta) | mAP@0.5:0.95, AP\_small, Recall por clase |
| MobileNetV3-S / EfficientNet-B0 | CNN clasificadora ligera | Segunda etapa: bus vs. camión y bicicleta vs. motocicleta | Accuracy, F1 por clase, matriz de confusión 2×2 |
| RT-DETR (opcional) | Transformer (detección sin NMS) | Las 6 clases; contraste arquitectónico | mAP@0.5, F1 macro, fps |
| ByteTrack / DeepSORT | Asociación (Kalman \+ IoU / Re-ID) | Todas las clases; mantenimiento de identidad | MOTA, IDF1, ID switches, MOTP, HOTA |

# **9\. Métricas de desempeño**

Las métricas se organizan en cuatro niveles: clasificación, detección, seguimiento y operación. Se fijan metas realistas para un prototipo académico con conjunto de datos reducido; alcanzarlas es un buen resultado, superarlas ampliamente debería levantar sospecha de fuga de datos.

## **9.1 Métricas de clasificación multiclase**

| Métrica | Qué mide y por qué importa aquí | Meta del prototipo |
| ----- | :---: | :---: |
| Exactitud (Accuracy) | Proporción global de aciertos. Engañosa con clases desbalanceadas: útil solo como cifra de contexto | ≥ 0,75 |
| Precisión (por clase) | De lo que el modelo llamó “bus”, cuánto era bus. Penaliza falsas alarmas en el reporte | ≥ 0,70 |
| Exhaustividad (Recall) | De todos los buses reales, cuántos encontró. Es la métrica crítica para conteos de aforo | ≥ 0,70 |
| F1 por clase | Media armónica de precisión y recall; revela las clases débiles | ≥ 0,65 en cada clase |
| F1 macro | Promedio simple del F1 de las seis clases; da el mismo peso a las minoritarias | ≥ 0,70 |
| Exactitud balanceada | Promedio de recall por clase; corrige el sesgo del desbalance | ≥ 0,70 |
| Matriz de confusión | Diagnóstico: no da un número, muestra exactamente qué se confunde con qué | Análisis obligatorio |

## **9.2 Métricas de detección**

| Métrica | Qué mide | Meta del prototipo |
| ----- | :---: | :---: |
| IoU | Solapamiento entre caja predicha y caja real; umbral de validez (0,5) | Umbral 0,5 |
| mAP@0.5 | Precisión promedio a IoU 0,5, promediada sobre clases. Métrica estándar de detección | ≥ 0,60 |
| mAP@0.5:0.95 | Promedio sobre umbrales de IoU de 0,5 a 0,95; mucho más exigente en la localización | ≥ 0,35 |
| Curva Precisión–Recall | Muestra el compromiso al mover el umbral de confianza; permite elegir el punto de operación | Reportar por clase |

## **9.3 Métricas de seguimiento multiobjeto**

| Métrica | Qué mide | Meta del prototipo |
| ----- | :---: | :---: |
| MOTA | Exactitud global del tracking: combina falsos positivos, falsos negativos y cambios de ID | ≥ 0,50 |
| MOTP | Precisión de la localización de los objetos correctamente seguidos | ≥ 0,70 |
| IDF1 | F1 sobre la identidad: qué tan bien se conserva el mismo ID a lo largo del tiempo | ≥ 0,55 |
| ID Switches | Número de veces que un objeto cambia de identificador. Cada switch daña el conteo | \< 10 por clip |
| HOTA | Métrica moderna que equilibra detección y asociación en un solo número | Reportar (referencia) |

## **9.4 Métricas operativas**

| Métrica | Qué mide | Meta del prototipo |
| ----- | :---: | :---: |
| FPS (cuadros por segundo) | Viabilidad de procesamiento en tiempo real | ≥ 15 fps en GPU de Colab |
| Latencia por cuadro | Tiempo de inferencia \+ asociación | ≤ 65 ms |
| Error de conteo (%) | Diferencia entre el conteo automático y el conteo manual de referencia | ≤ 15 % |
| Tiempo de generación del reporte | Desde el fin del procesamiento hasta el archivo exportado | ≤ 5 s |

# **10\. Matriz de confusión y análisis de errores**

La matriz de confusión es el instrumento central del análisis: la diagonal contiene los aciertos y cada celda fuera de la diagonal indica exactamente qué clase se está confundiendo con cuál. A continuación se presenta una matriz ilustrativa, construida sobre un conjunto de prueba hipotético de 240 instancias, con el formato y los valores plausibles que se espera obtener. Las filas son la clase real y las columnas la clase predicha.

| Real \\ Predicho | Peatón | Bicicleta | Motociclt. | Automóvil | Bus | Camión | No detect. |
| ----- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Peatón** | **52** | 3 | 2 | 1 | 0 | 0 | 2 |
| **Bicicleta** | 4 | **28** | 6 | 1 | 0 | 0 | 1 |
| **Motocicleta** | 1 | 5 | **35** | 2 | 0 | 0 | 2 |
| **Automóvil** | 0 | 0 | 2 | **48** | 2 | 2 | 1 |
| **Bus** | 0 | 0 | 0 | 1 | **15** | 3 | 1 |
| **Camión** | 0 | 0 | 0 | 2 | 4 | **13** | 1 |

*Los valores en negrilla corresponden a la diagonal (aciertos). Total \= 240 instancias; aciertos \= 191\.*

## **10.1 Métricas derivadas de la matriz**

| Clase | Precisión | Recall | F1 | Soporte |
| ----- | :---: | :---: | :---: | :---: |
| Peatón | 0,912 | 0,867 | 0,889 | 60 |
| Bicicleta | 0,778 | 0,700 | 0,737 | 40 |
| Motocicleta | 0,778 | 0,778 | 0,778 | 45 |
| Automóvil | 0,873 | 0,873 | 0,873 | 55 |
| Bus | 0,714 | 0,750 | 0,732 | 20 |
| Camión | 0,722 | 0,650 | 0,684 | 20 |
| **Promedio macro** | **0,796** | **0,770** | **0,782** | **240** |
| **Exactitud global** | **0,796 (191/240)** | — | — | — |

## **10.2 Lectura de los modos de falla**

**Bicicleta ↔ Motocicleta (6 y 5 errores).** Es la confusión dominante. Ambas clases comparten silueta —una persona erguida sobre un vehículo de dos ruedas— y a distancia media la diferencia se reduce a texturas finas (motor, escape, casco). Mitigación: clasificador en cascada (sección 6.5), recortes de mayor resolución y aumento del número de ejemplos de bicicleta.

**Bus ↔ Camión (3 y 4 errores).** Segunda confusión más frecuente, agravada por el bajo soporte de ambas clases (20 instancias cada una). Son objetos grandes y rectangulares que, vistos de frente o parcialmente ocluidos, se distinguen sobre todo por la disposición de ventanas. Mitigación: ponderación de clases, aumento dirigido y, si el soporte no mejora, colapsar a “vehículo pesado”.

**Peatón ↔ Bicicleta (3 y 4 errores).** Ocurre cuando el ciclista está detenido o cuando la bicicleta queda ocluida por otro objeto y solo es visible el torso. Mitigación: criterio de anotación explícito (ciclista y bicicleta forman una sola caja) y coherencia temporal aportada por el tracker.

**Columna “No detectado” (8 falsos negativos).** Objetos que el modelo no encontró en absoluto. Se concentran en instancias pequeñas, lejanas o fuertemente ocluidas. Mitigación: bajar el umbral de confianza (a costa de precisión), inferencia por mosaico (SAHI) para objetos pequeños, o recurrir a Faster R-CNN, mejor en ese régimen.

**Clases fuertes.** Peatón y Automóvil superan 0,87 de F1, algo esperable: son las clases con más ejemplos, más representadas en COCO y visualmente más distintivas. El resultado confirma que el desempeño del sistema está gobernado por el soporte de cada clase, no por la arquitectura.

## **10.3 Visualizaciones complementarias**

* Matriz de confusión normalizada por fila (porcentajes), para comparar clases con soportes muy distintos.

* Curvas Precisión–Recall por clase y curva F1 frente al umbral de confianza.

* Galería de los errores más confiados (highest-confidence mistakes): los casos donde el modelo se equivocó estando seguro suelen revelar problemas de anotación.

* Comparativa de barras de F1 por clase entre los tres modelos evaluados.

# **11\. Generación automática de reportes**

El módulo de reportes consume la salida del rastreador (lista de trayectorias con ID, clase y coordenadas por cuadro) y produce un documento estructurado sin intervención humana.

## **11.1 Contenido del reporte**

* Conteo total y desagregado por cada una de las seis clases.

* Flujo por línea virtual: cruces contabilizados por sentido (entrada/salida, norte-sur/sur-norte).

* Distribución temporal: conteos por franja horaria y detección de la hora pico del punto observado.

* Composición del tráfico: porcentaje de participación de cada clase, insumo directo para planeación de movilidad.

* Tiempo de permanencia por objeto en la zona de interés, con detección de permanencias anómalas.

* Alertas por reglas simples: circulación en contraflujo, presencia de camión en horario restringido, peatón sobre calzada, aglomeración por encima de un umbral.

* Anexo de calidad: número de objetos seguidos, ID switches y confianza media, para que quien lea el reporte sepa cuánto confiar en él.

## **11.2 Formato y arquitectura**

Salida en CSV (datos crudos para análisis posterior), PDF (informe legible con tablas y gráficas) y opcionalmente un tablero ligero en Streamlit. Implementación en Python con pandas para la agregación, matplotlib para las gráficas y ReportLab o WeasyPrint para el PDF, manteniendo un pipeline modular de un solo punto de entrada por línea de comandos.

# **12\. Metodología y cronograma**

| Fase | Actividades | Duración |
| ----- | :---: | :---: |
| 1\. Revisión | Estado del arte en MOT, selección de frameworks, definición del esquema de clases | Semanas 1–2 |
| 2\. Campo | Identificación de puntos, permisos, captura de los 50–100 clips en distintas franjas | Semanas 3–4 |
| 3\. Anotación | Etiquetado en CVAT, anonimización, control de calidad cruzado, partición por clip | Semanas 5–6 |
| 4\. Modelado | Fine-tuning de YOLO, Faster R-CNN y clasificador ligero; validación cruzada k=5 | Semanas 7–9 |
| 5\. Tracking | Integración de ByteTrack/DeepSORT, ajuste de umbrales, conteo por línea virtual | Semanas 10–11 |
| 6\. Evaluación | Cálculo de todas las métricas, matriz de confusión, análisis de errores, iteración | Semanas 12–13 |
| 7\. Reportes | Módulo de agregación, plantillas de PDF/CSV, reglas de alerta | Semana 14 |
| 8\. Cierre | Documento final, sustentación, repositorio y video demostrativo | Semanas 15–16 |

# **13\. Entregables**

* Conjunto de datos anotado y anonimizado (50–100 clips) con su ficha de documentación (datasheet).

* Repositorio con el código del pipeline completo: entrenamiento, inferencia, tracking y reportes.

* Modelos entrenados (pesos) de las tres arquitecturas evaluadas.

* Informe de evaluación con todas las métricas, matrices de confusión y análisis de errores.

* Reportes de ejemplo generados automáticamente en CSV y PDF.

* Documento final del trabajo de campo y video demostrativo de 3 a 5 minutos.

# **14\. Riesgos y consideraciones éticas**

| Riesgo | Impacto | Mitigación |
| ----- | :---: | :---: |
| Conjunto de datos insuficiente para las clases minoritarias | Métricas no interpretables en bus y camión | Captura dirigida en corredores de carga; colapso a 4 clases si persiste |
| Sobreajuste por pocos datos | Métricas infladas que no generalizan | Partición por clip, validación cruzada k=5, aumento agresivo |
| Condiciones adversas (lluvia, contraluz) | Caída del recall | Documentar como limitación; capturar en franjas variadas |
| Limitaciones de cómputo en Colab | Sesiones interrumpidas, modelos grandes inviables | Priorizar variantes nano; guardar puntos de control frecuentes |
| Tratamiento de datos personales | Riesgo legal y ético | Anonimización de rostros y placas, señalización del punto, no almacenar video crudo identificable, cumplimiento de la Ley 1581 de 2012 |
| Sesgo del sistema | Reportes que sobrerrepresentan ciertos actores | Reportar métricas por clase y no solo globales; declarar el desbalance explícitamente |

# **15\. Criterios de éxito**

El proyecto se considerará exitoso si, sobre el conjunto de prueba, se alcanza un F1 macro igual o superior a 0,70 con ninguna clase por debajo de 0,60; un mAP@0.5 igual o superior a 0,60; un MOTA igual o superior a 0,50 con IDF1 sobre 0,55; un error de conteo automático frente al conteo manual menor o igual al 15 %; y si el sistema genera el reporte completo sin intervención humana a partir de un clip de entrada. De forma igualmente importante, se espera un análisis de errores que explique los modos de falla observados, más allá de reportar las cifras.

# **16\. Referencias sugeridas**

* Zhang, Y. et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV.

* Jocher, G. et al. Ultralytics YOLO — documentación oficial y guías de entrenamiento personalizado.

* Ren, S. et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS.

* Bernardin, K. & Stiefelhagen, R. (2008). Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.

* Luiten, J. et al. (2021). HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking. IJCV.

* Howard, A. et al. (2019). Searching for MobileNetV3. ICCV.

* Congreso de Colombia (2012). Ley Estatutaria 1581 de 2012 — Protección de Datos Personales.