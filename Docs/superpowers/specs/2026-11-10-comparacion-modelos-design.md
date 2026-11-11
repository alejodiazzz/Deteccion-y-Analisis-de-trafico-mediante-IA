# Comparación de modelos en el tablero — diseño

Fecha: 2026-11-10 · Estado: pendiente de revisión

## Objetivo

Después de elegir un video y un punto de cámara en el tablero, comparar ahí mismo YOLO26n, YOLO26s, RT-DETR-L y Faster R-CNN, mostrar sus métricas y la matriz de confusión de cada uno cuando existan anotaciones, e indicar cuál fue el mejor **en ese video**.

## Principio rector

El tablero solo afirma lo que los datos permiten afirmar. Un video sin referencia humana no tiene matriz de confusión ni ganador por precisión.

| Datos disponibles | Se muestra | Ganador |
|---|---|---|
| Solo video | Cruces por clase, objetos seguidos, FPS de inferencia y discrepancia entre modelos | Ninguno; solo se indica el más rápido |
| Video + conteo manual | Lo anterior + error de conteo (MAE, RMSE y error por clase) | Menor MAE: "mejor contando" |
| Video + anotaciones CVAT | Lo anterior + mAP50, mAP50-95, F1 macro, HOTA, MOTA, IDF1, ID switches y matrices de confusión | Mayor F1 macro: "mejor detectando y clasificando" |

Con anotaciones y conteo a la vez se muestran ambos resultados por separado, sin combinarlos en una puntuación.

## Flujo de usuario

1. El usuario elige video y punto de cámara en el panel lateral existente.
2. En la pestaña nueva "Comparar modelos" puede añadir, opcionalmente, el zip CVAT MOT 1.1 y el CSV de conteo manual.
3. La pantalla indica qué nivel de resultados permitirán esos datos, qué falta para el siguiente y el tiempo estimado.
4. Al presionar "Comparar" se ejecutan los modelos uno por uno, con avance por modelo.
5. Se muestran el resultado destacado, la tabla, las barras por métrica, las matrices (si hay anotaciones) y las descargas.

La comparación no depende de haber presionado antes "Analizar".

## Cálculo

- Cada modelo se ejecuta a través de `CachedDetector` con confianza 0,01, igual que `evaluate`, de modo que una sola inferencia alimenta el conteo y la evaluación. La caché vive en `outputs/dashboard/<video>/comparison/detections/`.
- Conteo: `process_video` con el mismo tracker (ByteTrack) para todos y el umbral por defecto de cada modelo (`default_track_thresh`). El tracker descarta por sí mismo las detecciones por debajo de 0,1, como en el análisis normal.
- Evaluación: `evaluate` con confianza 0,25 para la matriz e IoU 0,5, y los mismos `coco_map` y `mot_metrics` que la línea de comandos.
- FPS de inferencia: el registrado por `CachedDetector` en la ejecución que generó la caché.
- Un fallo en un modelo no detiene a los demás: se muestra el error en su fila y ese modelo queda fuera de la selección del ganador.

## Discrepancia entre modelos (sin referencia)

Para cada clase, el rango de cruces entre modelos (máximo − mínimo) y el coeficiente de variación del total. Se presenta como desacuerdo, nunca como error.

## Reglas para declarar el mejor

Función pura, sin dependencias de modelos ni de interfaz:

- `best_by_annotations`: mayor F1 macro. Empatan los modelos a 0,01 o menos del mejor.
- `best_by_counts`: menor MAE. Empatan los modelos a menos de 0,5 cruces del mejor.
- `fastest`: mayor FPS de inferencia. Es un dato de velocidad, no un ganador global.
- Sin modelos válidos, el resultado es "sin ganador".

Los resultados se rotulan "en este video" y la pantalla recuerda que un solo clip no permite generalizar y que los umbrales no están calibrados.

## Conteo manual

CSV con columnas obligatorias `class` y `manual`, y opcionales `line` y `direction`.

- Las clases usan los identificadores internos (`pedestrian`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`).
- Todas las filas deben indicar `line` y `direction`, o ninguna. Sin ellas se compara el total por clase.
- Son errores: filas duplicadas, mezcla de granularidades, una línea o sentido inexistentes en el punto de cámara.
- Una columna `automatic` se ignora, porque el valor automático lo aporta cada modelo.
- El cálculo reutiliza `evaluate_count_rows`.

## Protecciones

- Si `data/splits.csv` existe y asigna el clip a `test`, se bloquea la comparación con anotaciones y se explica por qué: elegir un modelo con la partición de prueba invalida esa evaluación. Sigue disponible la comparación sin anotaciones.
- Si las anotaciones no coinciden con el video (número de cuadros o formato), se muestra el error de `read_annotations` o `evaluate`, sin resultados parciales de evaluación.

## Visualización

- Resultado destacado como texto y métricas, no como gráfico.
- Tabla comparativa como vista principal y accesible de todos los valores.
- Barras horizontales separadas por métrica; nunca dos escalas en un mismo eje. Sin gráficos de radar.
- Cada modelo conserva su orden fijo en todas las vistas y su nombre siempre aparece como etiqueta. Las barras usan un solo tono y resaltan al mejor con uno más claro, como el intervalo pico del tablero: una paleta de 4 colores no superó la validación de separación contra el fondo oscuro (naranja y amarillo, ΔE 10,6).
- Matrices de confusión: 4 pequeñas en una cuadrícula de 2 × 2, con la misma escala secuencial de un solo tono (0–100 %).
  - Filas: clases reales + "falso positivo". Columnas: clases predichas + "no detectado".
  - El color es el porcentaje de la fila. Cada celda muestra el número exacto y el tooltip añade porcentaje, clase real y predicha.
  - La celda "falso positivo × no detectado" no existe y queda vacía. Una fila sin objetos se muestra sin color y marcada "sin datos".
  - El texto de cada celda elige claro u oscuro según el fondo para mantener el contraste.

## Salidas

En `outputs/dashboard/<video>/comparison/`: `summary.csv`, y por modelo `confusion.csv` y `per_class.csv` (si hay anotaciones). Se descargan desde la pestaña. Cada comparación borra los resultados previos de esa carpeta, salvo la caché de detecciones, para no mezclar ejecuciones.

## Componentes

| Archivo | Responsabilidad |
|---|---|
| `vtia/domain/model_selection.py` (nuevo) | Reglas de ganador, empates y discrepancia |
| `vtia/application/compare_models.py` (nuevo) | Orquesta los modelos y devuelve un resultado por modelo; recibe sus dependencias, sin cargar modelos |
| `vtia/adapters/manual_counts.py` (nuevo) | Lee y valida el CSV de conteo manual |
| `vtia/adapters/comparison_files.py` (nuevo) | Escribe los CSV; recibe `_write_model_tables`, que hoy está en `cli.py` |
| `vtia/compose.py` | `compare_file()`: construye detectores, caché y tracker |
| `app/comparison_view.py` (nuevo) | Pestaña de Streamlit y gráficos Altair, fuera de `dashboard.py` |
| `app/dashboard.py` | Añade la pestaña y delega en `comparison_view` |

## Fuera de alcance

Ranking ponderado, medición de RAM, videos lado a lado, informe PDF de la comparación y calibración automática de umbrales. No se reutilizan las cifras escritas a mano de la pantalla de comparación de `repo_study`.

## Pruebas

- Reglas de selección: ganador, empates en el margen, sin modelos válidos, un modelo fallido.
- CSV de conteo manual: formatos válidos y cada error.
- `compare_models` con detectores y tracker simulados: los tres niveles, fallo de un modelo y bloqueo de la partición de prueba.
- Datos de la matriz: porcentaje por fila, celda inexistente y fila sin objetos.
- Prueba real: tablero con `streamlit.testing` y con los clips de Bogotá, que solo cubren el nivel sin anotaciones. Los niveles con referencia se verifican con datos sintéticos hasta contar con un clip anotado.

## Costo esperado

Con la GPU AMD (65, 37, 23 y 13 fps documentados), unos 2,5 minutos de detección para un clip de 30 s y unas 2,4 horas para un video de 30 min. En CPU es bastante más lento. La primera ejecución descarga unos 250 MB de pesos.
