# Guía de evaluación científica VTIA

## Alcance

La evaluación sobre datos propios utiliza videos anonimizados y exportaciones CVAT MOT 1.1 con el mismo nombre base. La evaluación no ejecuta inferencia hasta que se solicita `evaluate`; `validate-dataset` solo inspecciona integridad e inventario.

Los resultados de VisDrone2019-MOT y los de cámaras propias deben conservarse como experimentos distintos. No se deben combinar en un mismo agregado sin identificar el conjunto, la partición, los pesos y la configuración.

## 1. Preparar y validar el conjunto

```powershell
python -m vtia.cli validate-dataset --videos data\anon --cvat data\annotations --out outputs\dataset_quality.json
```

El informe JSON incluye videos disponibles, anotados y pendientes, cuadros con anotaciones, objetos por clase y secuencia, distribución por punto y franja horaria, regiones ignore, anotaciones huérfanas, nombres duplicados y secuencias inválidas. Un video sin anotación se informa como pendiente; no se inventan etiquetas.

Los nombres de clips deben seguir `P<point>_<YYYYMMDD>_<HHMM>[_s<SSSS>]`. Las exportaciones MOT deben incluir exactamente un `gt/gt.txt` y un `gt/labels.txt`, con nueve columnas por fila, frames MOT de base 1, clases conocidas y cajas válidas.

## 2. Crear particiones

```powershell
python -m vtia.cli split data\anon --out data\splits.csv --test-share 0.3 --folds 5 --seed 0
```

Se genera `data/splits.csv` y su manifiesto `data/splits.manifest.json`. La asignación mantiene todos los clips de una sesión de captura en una sola partición. La semilla y los parámetros quedan registrados. El conjunto `test` es fijo; `fold1` a `fold5` son grupos de validación cruzada.

No modificar manualmente el CSV después de generarlo. Si cambia el conjunto de clips, regenerar el manifiesto y documentar el nuevo experimento.

## 3. Evaluar el conjunto de prueba

```powershell
python -m vtia.cli evaluate --cvat data\annotations --videos data\anon --splits data\splits.csv --partition test
```

La evaluación de datos propios exige declarar `--splits` y `--partition`. Los resultados se escriben en CSV, JSON, Markdown y PDF. Se incluyen el origen del conjunto, la partición, las secuencias, el dispositivo, el tracker y los parámetros de evaluación.

Para evaluar un fold durante validación cruzada, se puede seleccionar `--partition fold1` (o fold2…fold5). El conjunto de prueba no se utiliza para calibración.

## 4. Calibrar solo con validación

```powershell
python -m vtia.cli calibrate --cvat data\annotations --videos data\anon --splits data\splits.csv --partition fold1 --model yolo26n.pt
```

La partición `test` está prohibida para `calibrate`. Los umbrales elegidos con folds deben registrarse y quedar congelados antes de la evaluación final sobre `test`.

## 5. Ajuste fino

```powershell
python -m vtia.cli train --cvat data\annotations --videos data\anon --splits data\splits.csv --fold 1 --name fold1
```

El fold seleccionado se usa como validación; los demás folds se usan para entrenamiento. El conjunto `test` queda excluido. No se debe entrenar con el conjunto de prueba ni seleccionar hiperparámetros con él.

## 6. Evaluación del error de conteo

Crear un CSV de registros manuales revisados y conteos automáticos correspondientes. Columnas obligatorias: `manual`, `automatic`, `class`. La clase usa los identificadores internos `pedestrian`, `bicycle`, `motorcycle`, `car`, `bus` o `truck`. Columnas opcionales: `video`, `point`, `date`, `time_band`, `direction`, `line`.

Cada fila representa una unidad de comparación coherente (por ejemplo, un video × clase × línea × sentido). No mezclar totales por video con eventos por línea en la misma agregación.

```csv
video,point,date,time_band,class,direction,line,manual,automatic
P1_20261015_0705_s0312,P1,2026-10-15,morning,car,in,line_1,24,22
```

```powershell
python -m vtia.cli evaluate-counting --csv data\manual_counts.csv --out outputs\counting_eval
```

Se generan JSON, CSV y Markdown con diferencia firmada, diferencia absoluta, error porcentual, MAE y RMSE, además de desgloses por clase, video, punto, franja, dirección y línea. Si el conteo manual es cero, el error porcentual se marca como `null` (N/D); la diferencia absoluta se conserva.

Los conteos manuales deben proceder de revisión humana/anotación y los automáticos de una ejecución VTIA identificable. El evaluador no genera ni estima conteos reales.

## Limitaciones conocidas

- La validación CVAT comprueba formato, clases, IDs, duplicados, cajas y rango de cuadros; no certifica la calidad semántica de cada etiqueta ni la efectividad de la anonimización.
- La distribución temporal depende de que los nombres cumplan el protocolo de captura.
- La evaluación del error de conteo requiere un CSV de conteos manuales y automáticos comparables.
- Los indicadores de memoria GPU y robustez por iluminación/oclusiones no se generan todavía de forma automática.

## 7. Análisis de robustez por condiciones

La evaluación exporta métricas por secuencia en `per_sequence.csv` y conserva los valores completos en `summary.json`. Para agruparlas por condiciones registradas durante la captura, preparar un CSV con una fila por secuencia y las columnas:

`sequence,lighting,occlusion,vehicle_density,pedestrian_density,object_size,distance`

Los valores son etiquetas descriptivas revisadas por el equipo (por ejemplo, `good`, `low`, `backlight`; `low`, `medium`, `high`). No se asignan automáticamente a partir del video. Ejecutar:

```powershell
python -m vtia.cli evaluate --cvat data\annotations --videos data\anon --splits data\splits.csv --partition test --conditions data\conditions.csv
```

El archivo `robustness*.json` resume mAP50, mAP50-95, F1 macro, HOTA, MOTA e IDF1 por condición. Se exigen al menos tres secuencias para declarar evidencia; con menos, el resultado queda explícitamente como `insufficient_evidence`. El umbral de tres es un mínimo operativo, no una garantía de potencia estadística.

## 8. Archivos de resultados de evaluación

- `summary*.csv`, `summary*.json`, `summary*.md`, `summary*.pdf`: consolidado por modelo.
- `<modelo>/per_class.csv`: métricas por clase.
- `<modelo>/confusion.csv`: matriz de confusión.
- `<modelo>/per_sequence.csv`: métricas por secuencia.
- `robustness*.json`: resumen por condición, solo si se suministra `--conditions`.

En la tabla consolidada se distinguen `fps_inference` (inferencia del detector) y `fps_processing` (pipeline completo de evaluación). El número de parámetros y el tamaño del checkpoint se informan cuando pueden medirse; los recursos no disponibles se dejan vacíos/N/D. La memoria RAM/GPU no se mide aún de forma portable en todos los entornos.
