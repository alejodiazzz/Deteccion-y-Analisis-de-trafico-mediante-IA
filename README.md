# VTIA — Video tracking e IA urbana

Prototipo que detecta, sigue y cuenta actores viales (peatón, bicicleta, motocicleta, automóvil, bus, camión) en video de cámara fija y genera reportes.

- Plan: [`Docs/Plan_de_Accion_VideoTracking.md`](Docs/Plan_de_Accion_VideoTracking.md)
- Avance: [`Docs/AVANCE.md`](Docs/AVANCE.md)

## Instalación (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Video de ejemplo (35 MB):

```powershell
mkdir data\samples
curl.exe -Lo data\samples\vehicles.mp4 https://media.roboflow.com/supervision/video-examples/vehicles.mp4
```

## Uso

### Tablero (lo más fácil)

```powershell
python -m streamlit run app\dashboard.py
```

Se abre en el navegador (http://localhost:8501). Solo es accesible desde el propio PC (`.streamlit/config.toml`), porque los videos pueden tener datos personales. El fondo animado es [upstream-fx](https://github.com/AmunM9/upstream-fx) (MIT), incluido en `app/static/` para no depender de internet. Ahí se sube o elige un video, se elige el punto de cámara y el modelo, y se presiona **Analizar**. Muestra conteos, gráficas, el video anotado, zonas y alertas, y permite descargar todo (PDF y CSV). Los resultados quedan en `outputs\dashboard\<video>\`.

### Comparar modelos

En el tablero, la pestaña **Comparar modelos** ejecuta YOLO26n, YOLO26s, RT-DETR-L y Faster R-CNN sobre el video y el punto elegidos. Solo afirma lo que los datos permiten:

| Además del video | Resultado |
|---|---|
| Nada | Cruces, objetos seguidos, velocidad y discrepancia entre modelos, sin ganador por precisión |
| Conteo manual (CSV) | Error de conteo por modelo y el mejor contando |
| Anotaciones CVAT (zip MOT 1.1) | Matriz de confusión de cada modelo, mAP, F1, HOTA, MOTA, IDF1 y el mejor detectando y clasificando |

El CSV lleva las columnas `class` (`pedestrian`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`) y `manual`, y opcionalmente `line` y `direction` en todas sus filas. Un clip de la partición de prueba de `data/splits.csv` no se puede comparar con anotaciones, porque elegir un modelo con él invalidaría esa evaluación. El resultado vale para ese video y con umbrales sin calibrar. Los archivos quedan en `outputs\dashboard\<video>\comparison\`, y las detecciones se guardan en caché, así que repetir una comparación es casi inmediato.

### Consola

```powershell
python -m vtia.cli count data\samples\vehicles.mp4 --point configs\points\sample_vehicles.yaml --save-video
```

Escribe en `outputs\<video>\`:

| Archivo | Contenido |
|---|---|
| `report.pdf` | Reporte legible: conteos, composición, distribución temporal y pico, permanencia, alertas, anexo de calidad |
| `annotated.mp4` | Con `--save-video`: cajas por clase con ID, zonas, línea que se ilumina en cada cruce, conteos y alertas |
| `events.csv` | Un cruce por fila (cuadro, segundo, hora real si se conoce, ID, clase, línea, sentido) |
| `counts.csv`, `by_class.csv`, `by_interval.csv` | Totales por línea/sentido/clase, por clase y por intervalo de tiempo |
| `visits.csv`, `dwell.csv` | Visitas a cada zona y su permanencia |
| `alerts.csv` | Alertas: contraflujo, horario restringido, clase no permitida en zona, permanencia prolongada, aglomeración |

Opciones:
- `--model yolo26s.pt | rtdetr-l.pt | faster_rcnn` (por defecto `yolo26n.pt`)
- `--start "2026-10-15 07:05"`: hora real de inicio, si el nombre del archivo no sigue el protocolo (`P1_20261015_0705.mp4`)
- `--tracker botsort`, `--track-thresh 0.3` (por defecto 0,3 para yolo26n y 0,6 para los demás)
- `--dedup 0.7` (elimina cajas duplicadas de cualquier clase; apagado por defecto)
- `--conf 0.3` (confianza mínima del detector), `--out carpeta`

**Punto de cámara:** cada punto necesita su YAML en `configs/points/` con líneas (y opcionalmente sentido permitido), zonas con sus reglas y restricciones horarias. Ver los dos ejemplos comentados: `sample_vehicles.yaml` y `sample_people.yaml` (para `data\samples\people-walking.mp4`).

### Preparar los datos propios (Fase 2)

Flujo completo, de la grabación a la evaluación (los nombres siguen el protocolo: `P1_20261015_0705_s0312.mp4`):

```powershell
# 1. Anonimizar (en el PC servidor, antes de que nada salga de él)
python -m vtia.cli anonymize data\raw --out data\anon
#    -> revisar a ojo cada data\anon\*_check.jpg y los videos; recién ahí borrar data\raw

# 2. Partición por sesión: prueba fija + 5 grupos para validación cruzada
python -m vtia.cli split data\anon --out data\splits.csv

# 3. Pre-anotación para CVAT (los anotadores solo corrigen)
python -m vtia.cli prelabel data\anon --out data\prelabels
#    -> en CVAT: crear una tarea por clip con el video anonimizado,
#       Actions > Upload annotations > MOT 1.1 > data\prelabels\<clip>.zip
#    -> corregir y exportar: Actions > Export task dataset > MOT 1.1 -> data\annotations\<clip>.zip

# 4. Validar inventario y anotaciones (no ejecuta inferencia)
python -m vtia.cli validate-dataset --videos data\anon --cvat data\annotations --out outputs\dataset_quality.json

# 5. Evaluar exclusivamente la prueba fija
python -m vtia.cli evaluate --cvat data\annotations --videos data\anon --splits data\splits.csv --partition test

# 6. Calibrar únicamente con un fold de validación (nunca con test)
python -m vtia.cli calibrate --cvat data\annotations --videos data\anon --splits data\splits.csv --partition fold1 --model yolo26n.pt

# 7. Ajuste fino: valida con fold1, entrena con los demás folds, excluye test
python -m vtia.cli train --cvat data\annotations --videos data\anon --splits data\splits.csv --fold 1 --name fold1
# Evaluar los pesos finales en la prueba fija
python -m vtia.cli evaluate --cvat data\annotations --videos data\anon --splits data\splits.csv --partition test --models outputs\train\fold1\weights\best.pt yolo26n.pt

# 8. Evaluar conteos manuales frente a conteos automáticos
python -m vtia.cli evaluate-counting --csv data\manual_counts.csv --out outputs\counting_eval
```

- La guía completa de validación de datos, particiones, evaluación y error de conteo está en [`Docs/Guia_Evaluacion_Cientifica.md`](Docs/Guia_Evaluacion_Cientifica.md).
- **Anonimización:** difumina la cabeza de cada persona, la franja de la placa de cada vehículo y el parabrisas de los vehículos grandes (≥ 250 px de alto), usando el mismo detector (YOLO26s, confianza 0,15). No guarda el audio. Los parámetros están en `vtia/domain/privacy.py`.
- **Pre-anotación:** usa YOLO26x (fuera de la comparación, para no sesgar la referencia) y une jinete + vehículo en una sola caja, como pide la guía.
- **Regla del jinete (E2):** `--riders 0.4` en `count`, `evaluate` y `calibrate`.

### Evaluar contra VisDrone2019-MOT (val)

```powershell
python -m vtia.cli evaluate --visdrone data\visdrone\VisDrone2019-MOT-val
```

Corre YOLO26n, YOLO26s, RT-DETR-L y Faster R-CNN (cambiar con `--models`), guarda sus detecciones en `outputs\eval\visdrone\detections\` (la segunda vez las reutiliza) y escribe `summary.csv`, `summary.json`, `summary.md` y `summary.pdf`, además de métricas por secuencia, clase y matriz de confusión para cada modelo. Acepta `--tracker`, `--track-thresh`, `--dedup` y `--riders`; con `--sequences` se evalúa solo un subconjunto; con valores distintos a los por defecto, el resumen se guarda con un sufijo (p. ej. `summary_botsort.md`).

Video de ejemplo con personas (para probar zonas y alertas):

```powershell
curl.exe -Lo data\samples\people-walking.mp4 https://media.roboflow.com/supervision/video-examples/people-walking.mp4
```

## Pruebas

```powershell
python -m unittest discover -s tests
```

## GPU AMD (opcional)

> Entrenar en la GPU AMD funciona con MIOpen desactivado (`vtia/adapters/yolo_training.py` lo hace solo): ROCm 7.2.1 en Windows no compila su kernel de normalización por lotes para entrenamiento.

En el PC servidor (Radeon RX 9060 XT, Windows 11) hay un segundo entorno, `.venv-rocm`, con PyTorch para AMD (ROCm 7.2.1). Necesita driver Adrenalin 26.2.2 o más reciente. Para recrearlo:

```powershell
py -3.12 -m venv .venv-rocm
$R = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
.venv-rocm\Scripts\pip install --no-cache-dir $R/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl $R/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl $R/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl $R/rocm-7.2.1.tar.gz
.venv-rocm\Scripts\pip install --no-cache-dir "$R/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "$R/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
"torch==2.9.1+rocm7.2.1`ntorchvision==0.24.1+rocm7.2.1" | Out-File -Encoding ascii .venv-rocm\constraints.txt
.venv-rocm\Scripts\pip install -c .venv-rocm\constraints.txt -r requirements.txt
```

Se usa igual que el entorno normal, cambiando `python` por `.venv-rocm\Scripts\python`. Los modelos usan la GPU automáticamente. El archivo de restricciones evita que pip reemplace el PyTorch de AMD.

## Estructura

```
vtia/
├── domain/        reglas puras, sin librerías externas
│   ├── classes.py      6 clases + mapeo de etiquetas (COCO y modelos ajustados)
│   ├── models.py       Detection, GroundTruth, Line, Zone, PointConfig, Visit, Alert, ...
│   ├── counting.py     cruce de línea, sentido, clase por mayoría
│   ├── zones.py        quién está en cada zona y cuánto tiempo
│   ├── alerts.py       reglas de alerta
│   ├── timeline.py     hora real desde el nombre del archivo
│   ├── privacy.py      qué difuminar: cabezas, placas, parabrisas
│   ├── splits.py       partición prueba / grupos, por sesión de captura
│   ├── evaluation.py   IoU, regiones ignoradas, emparejamiento, matriz de confusión, P/R/F1
│   └── postprocess.py  eliminación de duplicados, regla del jinete (E2)
├── application/   casos de uso
│   ├── ports.py           interfaces Detector y Tracker
│   ├── process_video.py   analizar un video (conteo, zonas, alertas)
│   ├── summary.py         agregados para el reporte
│   ├── evaluate.py        evaluar contra un dataset anotado
│   ├── anonymize.py       regiones a difuminar por cuadro (con retención)
│   └── postprocessed.py   detector con un paso de limpieza (dedup, jinetes)
├── adapters/      todo lo que toca librerías o archivos
│   ├── ultralytics_detector.py, faster_rcnn_detector.py   detectores
│   ├── mot_tracker.py           ByteTrack / BoT-SORT (paquete trackers)
│   ├── cached_detector.py       guarda y reutiliza detecciones
│   ├── video_io.py              leer video; escribir H.264 con ffmpeg (imageio-ffmpeg)
│   ├── annotated_video.py, anonymized_video.py   video anotado / anonimizado (+ hoja de control)
│   ├── yaml_config.py           configs/points/*.yaml -> PointConfig
│   ├── cvat_mot.py              CVAT MOT 1.1: escribir pre-anotaciones, leer exportaciones
│   ├── yolo_training.py         dataset YOLO + ajuste fino (Fase 4)
│   ├── report_files.py, table_files.py   CSV y Markdown
│   ├── pdf_report.py, charts.py, labels_es.py   reporte PDF
│   └── visdrone.py, coco_map.py, trackeval_metrics.py   evaluación
├── compose.py     arma los adaptadores y corre los casos de uso (lo comparten consola y tablero)
└── cli.py         consola: count, anonymize, prelabel, split, evaluate, calibrate, train
app/dashboard.py   tablero Streamlit (app/static/: fondo animado upstream-fx, MIT)
```

Regla de dependencias: `domain` no importa nada del proyecto fuera de sí mismo; `application` solo importa `domain`; `adapters`, `compose.py` y los puntos de entrada pueden importar todo.

## CVAT (anotación)

Corre en el PC servidor, en `C:\Users\USER\cvat` (v2.78.0).

| Acción | Comando (en `C:\Users\USER\cvat`) |
|---|---|
| Encender | `docker compose up -d` |
| Apagar | `docker compose stop` |
| Crear administrador (una vez) | `docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'` |

Interfaz: http://localhost:8080

**Acceso del equipo por Tailscale:**
1. Los 3 instalan [Tailscale](https://tailscale.com/download) y entran a la misma red (el dueño del servidor invita a los otros dos).
2. En el PC servidor, crear `C:\Users\USER\cvat\.env` con la línea `CVAT_HOST=<IP de Tailscale del servidor>` y ejecutar `docker compose up -d`. Desde ese momento CVAT se abre en `http://<IP de Tailscale>:8080`, también en el propio servidor.
3. Si no carga desde otro PC, permitir el puerto 8080 en el Firewall de Windows del servidor.

**Etiquetas del proyecto** (en CVAT: Projects → + → Raw, pegar). `ignore` marca regiones que no se evalúan; ver `Docs/Guia_Anotacion.md`:

```json
[
  {"name": "pedestrian", "type": "rectangle", "attributes": []},
  {"name": "bicycle", "type": "rectangle", "attributes": []},
  {"name": "motorcycle", "type": "rectangle", "attributes": []},
  {"name": "car", "type": "rectangle", "attributes": []},
  {"name": "bus", "type": "rectangle", "attributes": []},
  {"name": "truck", "type": "rectangle", "attributes": []},
  {"name": "ignore", "type": "rectangle", "attributes": []}
]
```
