# Experimento: ajuste fino de YOLO26n con MIO-TCD

Fecha: 2026-11-11. Rama `feature/comparacion-modelos` (commits `ceec0b3`, `6bace8c` y `a537d89`). La bitácora general es [`AVANCE.md`](AVANCE.md).

---

## 1. Resumen

- **Objetivo.** Mientras no hay clips propios anotados, ajustar el detector con un conjunto público de cámaras de tráfico fijas, como paso previo al experimento E3 del plan (ajuste fino de YOLO nano).
- **Conjunto elegido:** MIO-TCD Localization (Miovision). Se descartó una copia publicada en Roboflow porque está **mal anotada**: tiene una sola caja por imagen.
- **Resultado en la validación de MIO-TCD:** con 20.000 imágenes y 30 épocas en una T4 de Colab, YOLO26n llega a **mAP50 0,866 y mAP50-95 0,632**. En vehículos supera 0,9 de mAP50; en peatones queda en **0,526**, con un recall de 0,45.
- **Usar los nombres de COCO** (`person` en vez de `pedestrian`), para que las 6 clases partan de los pesos preentrenados, **no mejoró a los peatones**. La debilidad viene de los datos: hay pocos peatones y son pequeños.
- **Estos números no comparan modelos ni prueban nada en las cámaras del proyecto.** La validación es una partición al azar de MIO-TCD, no la prueba fija propia (ver §7).
- **Modelo resultante:** `MyDrive/VTIA/train/mio_tcd_person/weights/best.pt`. Uso no comercial (CC BY-NC-SA 4.0).

---

## 2. Conjuntos considerados

| Conjunto | Qué tiene | Decisión |
|---|---|---|
| **MIO-TCD Localization oficial** ([página](https://tcd.miovision.com/challenge/dataset.html)) | ~110.000 imágenes de entrenamiento con todas sus cajas, de miles de cámaras de tráfico en Canadá y EE. UU. Las anotaciones de prueba no son públicas | **Usado** |
| MIO-TCD Classification | 648.959 recortes con una etiqueta por imagen | No sirve para un detector |
| Copia de MIO-TCD en Roboflow (`thesis-ltv3y/mio_tcd-f3kk4`) | 7.242 imágenes de entrenamiento con **una caja por imagen** | **Descartada** (§5.1) |
| UA-DETRAC ([página](https://sites.google.com/view/daweidu/projects/ua-detrac)) | 100 videos (más de 140.000 cuadros) con IDs de seguimiento; solo vehículos | No se usa para entrenar, porque peatones, bicicletas y motos aparecen sin anotar. Es útil más adelante para **evaluar** seguimiento y conteo de vehículos |

**Lo que MIO-TCD no permite:** son imágenes sueltas, sin video ni IDs. Sirve para el detector, pero no para medir HOTA, MOTA, IDF1 ni error de conteo.

**Licencia:** CC BY-NC-SA 4.0. Es solo para uso no comercial, hay que citar el artículo y **los pesos entrenados heredan esas condiciones**. Cita: Z. Luo et al., *MIO-TCD: A new benchmark dataset for vehicle classification and localization*, IEEE Transactions on Image Processing, 2018.

---

## 3. Herramientas que se agregaron

| Pieza | Qué hace |
|---|---|
| `vtia/adapters/mio_tcd.py` | `convert_official` lee `train/` + `gt_train.csv` (`id,clase,x1,y1,x2,y2`), recorta las cajas al borde, separa una validación al azar y escribe el dataset YOLO. `convert_export` lee una exportación de Roboflow |
| `python -m vtia.cli train --mio-tcd DIR [--sample N]` | Entrenamiento desde la consola. Acepta la versión oficial o una exportación de Roboflow |
| `notebooks/mio_tcd_colab.ipynb` | Colab: clona el repositorio, descarga el dataset oficial al disco local, convierte, entrena guardando los pesos en Drive en cada época y permite reanudar |
| `vtia/adapters/yolo_training.py` | El modelo ajustado usa los nombres de COCO (`person`, `bicycle`, ...) con los IDs en el orden de `RoadClass`. VTIA traduce `person` → `pedestrian` al leer las detecciones |
| `tests/test_mio_tcd.py` | 5 pruebas: traducción de clases, regiones ignoradas, recorte de cajas, muestreo, partición y nombres de COCO |

**Traducción de clases** (según [`Guia_Anotacion.md`](Guia_Anotacion.md)):

| MIO-TCD | VTIA |
|---|---|
| pedestrian | `pedestrian` |
| bicycle | `bicycle` |
| motorcycle | `motorcycle` |
| car, pickup_truck | `car` |
| bus | `bus` |
| articulated_truck, single_unit_truck | `truck` |
| work_van, motorized_vehicle, non-motorized_vehicle | región ignorada (se pinta de gris) |

*Work van* se ignora porque las furgonetas siguen como caso [ABIERTO] en la guía. Las otras dos no corresponden a ninguna clase del proyecto.

---

## 4. Configuración común

| | |
|---|---|
| Modelo de partida | `yolo26n.pt` (COCO) |
| Entrenamiento | Ultralytics 8.4.174, 30 épocas, `imgsz` 640, lote 32, AdamW (lr 0,001, elegido automáticamente), aumentos por defecto |
| Hardware | Colab, Tesla T4 (15 GB), Python 3.13, torch 2.14.1+cu130 |
| Métricas | mAP50, mAP50-95, precisión y recall por clase, sobre la validación de cada conjunto |

---

## 5. Experimentos

### 5.1 Copia de Roboflow (descartada)

7.242 imágenes de entrenamiento y 1.553 de validación con 1.410 cajas; 30 épocas en 55 minutos.

| Clase | Cajas en val. | mAP50 | mAP50-95 |
|---|---|---|---|
| pedestrian | 16 | 0,028 | 0,012 |
| bicycle | 14 | 0,433 | 0,246 |
| motorcycle | 7 | 0,421 | 0,228 |
| car | 1.223 | 0,555 | 0,396 |
| bus | 67 | 0,617 | 0,463 |
| truck | 83 | 0,407 | 0,287 |
| **Todas** | **1.410** | **0,410** | **0,272** |

**Diagnóstico.** En la validación, el número de cajas de cada clase coincidía exactamente con el número de imágenes en que aparecía. Ninguna imagen tenía dos objetos de la misma clase. Al revisar la descarga original:

- **Las 7.242 imágenes de entrenamiento tienen exactamente 1 caja.**
- **La caja mediana ocupa el 1,1 % de la imagen.** Son escenas completas, no recortes.

Es decir, son escenas con varios vehículos en las que se anotó uno solo. El entrenamiento castiga al modelo cuando detecta los demás, porque los cuenta como falsos positivos. Así aprende a detectar menos, lo que explica el recall de 0,44 y el abandono de la clase peatón (precisión 1, recall 0). **El error está en la copia, no en el conversor**, que conserva todas las líneas de cada archivo y lo verifica una prueba.

### 5.2 MIO-TCD oficial, clase `pedestrian`

Muestra de 20.000 imágenes al azar (semilla 0): 18.000 de entrenamiento (430 sin objetos) y 2.000 de validación (60 sin objetos), con **5.945 cajas en validación**, unas 3 por imagen. 30 épocas en 2,08 horas. Ultralytics reutilizó las filas de COCO de 5 de las 6 clases (`Remapped 5/6`): `pedestrian` no coincide con `person` y partió de cero.

### 5.3 MIO-TCD oficial, clase `person`

Mismo dataset, misma semilla y misma validación. Las 6 clases reutilizan las filas de COCO (`Remapped 6/6`). Colab cortó la sesión por límite de uso de GPU durante la época 29. Se usa `best.pt`, de la **época 24**, y se validó después en CPU. Las épocas que faltaban habrían aportado poco: en las dos corridas la curva se aplana desde la época 23.

### 5.4 Resultados por clase (validación de MIO-TCD, 2.000 imágenes)

| Clase | Cajas | mAP50 5.2 | mAP50 5.3 | mAP50-95 5.2 | mAP50-95 5.3 | Recall 5.2 | Recall 5.3 |
|---|---|---|---|---|---|---|---|
| pedestrian / person | 104 | 0,525 | 0,526 | 0,246 | 0,246 | 0,43 | 0,45 |
| bicycle | 35 | 0,912 | 0,924 | 0,626 | 0,640 | 0,89 | 0,89 |
| motorcycle | 28 | 0,886 | 0,892 | 0,581 | 0,584 | 0,77 | 0,79 |
| car | 5.253 | 0,969 | 0,972 | 0,724 | 0,726 | 0,92 | 0,91 |
| bus | 240 | 0,967 | 0,965 | 0,880 | 0,878 | 0,94 | 0,95 |
| truck | 285 | 0,908 | 0,917 | 0,712 | 0,721 | 0,84 | 0,84 |
| **Todas** | **5.945** | **0,861** | **0,866** | **0,628** | **0,632** | **0,80** | **0,80** |

### 5.5 Resumen de las tres corridas

| Corrida | Imágenes (entr. / val.) | mAP50 | mAP50-95 | Peatones mAP50 |
|---|---|---|---|---|
| 5.1 Roboflow | 7.242 / 1.553 | 0,410 | 0,272 | 0,028 |
| 5.2 Oficial, `pedestrian` | 18.000 / 2.000 | 0,861 | 0,628 | 0,525 |
| 5.3 Oficial, `person` | 18.000 / 2.000 | **0,866** | **0,632** | 0,526 |

Las corridas 5.2 y 5.3 se comparan directamente porque comparten la validación. La 5.1 se midió sobre otras imágenes y solo sirve de referencia.

---

## 6. Hallazgos

1. **Revisar la anotación antes de entrenar.** Una copia pública con el mismo nombre que un dataset conocido puede estar incompleta. Bastan dos comprobaciones: cajas por imagen y tamaño de las cajas.
2. **Con anotación completa, YOLO26n aprende muy bien los vehículos de cámaras fijas:** mAP50 entre 0,91 y 0,97 en auto, bus y camión.
3. **Los peatones siguen débiles (mAP50 0,53, recall 0,45)** y reutilizar la fila de `person` de COCO no lo cambia. Las causas probables están en los datos. Son pocos: alrededor del 2 % de las cajas (104 de 5.945 en validación). Y son pequeños, porque las cámaras de tráfico los ven de lejos.
4. **Bicicletas y motos tienen buenas métricas, pero sobre 35 y 28 cajas.** Una sola imagen mueve mucho el resultado, así que no se pueden dar por resueltas.
5. **Las diferencias entre 5.2 y 5.3 están dentro del ruido.** Se conserva 5.3 porque no es peor en ninguna clase y porque `person` es el nombre adecuado para cualquier ajuste futuro.

---

## 7. Limitaciones

- **La validación es optimista.** MIO-TCD no indica la cámara de cada imagen, y la partición al azar puede dejar cuadros casi iguales en entrenamiento y en validación.
- **No hay línea base comparable.** No se midió `yolo26n.pt` de COCO sobre esta misma validación, así que estos números no dicen cuánto mejoró el ajuste respecto del modelo preentrenado.
- **Otro dominio.** Son cámaras de Canadá y EE. UU., de baja resolución y de hace años. Un buen resultado aquí no garantiza uno bueno en los puntos de cámara del proyecto.
- **Una sola ejecución por configuración** (semilla 0), sin media ni desviación estándar.
- **Se usó una muestra** (20.000 de ~110.000 imágenes) por el tiempo de GPU disponible en Colab gratuito.
- **No se verificó la convención de MIO-TCD para bicicletas y motos.** La guía de VTIA pide una sola caja para conductor y vehículo; MIO-TCD podría anotar solo el vehículo. Hay que revisarlo en `val_batch*_labels.jpg`.

---

## 8. Próximos pasos

1. **Prueba fija propia (la que decide):**
   `evaluate --cvat ... --partition test --models <best.pt> yolo26n.pt`. Para peatones es posible que gane el modelo de COCO y para vehículos el de MIO-TCD. Ese resultado también es válido para el informe.
2. **Ajuste con datos propios partiendo de este modelo:**
   `train --cvat ... --splits data\splits.csv --fold 1 --weights <best.pt>`.
3. **Si los peatones siguen siendo un problema en datos propios:** probar con `IMGSZ = 960` o con más imágenes de MIO-TCD. Las dos opciones requieren más tiempo de GPU que el de Colab gratuito.
4. **UA-DETRAC como evaluación de seguimiento y conteo de vehículos** con cámara fija, con un lector parecido al de VisDrone.

---

## 9. Cómo reproducirlo

**En Colab:** abrir `notebooks/mio_tcd_colab.ipynb` desde GitHub (rama `feature/comparacion-modelos`), elegir GPU T4 y correr las celdas en orden. Parámetros usados: `SAMPLE = 20000`, `EPOCHS = 30`, `BATCH = 32`, `IMGSZ = 640`, `RUN_NAME = "mio_tcd_person"`.

**En el PC:**

```powershell
curl.exe -Lo data\MIO-TCD-Localization.tar https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar
tar -xf data\MIO-TCD-Localization.tar -C data
python -m vtia.cli train --mio-tcd data\MIO-TCD-Localization --sample 20000 --name mio_tcd_person --epochs 30 --batch 32
```

**Resultados en Drive (`MyDrive/VTIA/train/`):**
- `mio_tcd/`: corrida 5.2, con pesos, curvas, matrices de confusión y `results.csv`.
- `mio_tcd_person/weights/`: corrida 5.3, con `best.pt` (época 24) y `last.pt` (época 28).

**Problemas encontrados en Colab y cómo se resolvieron:**
- **Conflicto de pandas.** `requirements.txt` instalaba pandas 3, que choca con el de Colab. El cuaderno ahora omite pandas, streamlit y altair, que solo usa el tablero.
- **`No module named 'vtia'`.** Aparece al cambiar de carpeta con `%cd`, o cuando los cambios no se han subido a GitHub. La celda de conversión vuelve a `/content/vtia`.
- **Python sigue usando código viejo después de `git pull`.** Hay que reiniciar la sesión. Para el cambio a `person`, se comprueba en el registro con la línea `Remapped 6/6`.
- **Límite de GPU de Colab gratuito.** Los pesos se guardan en Drive en cada época, y la celda 7b reanuda desde `last.pt`.
