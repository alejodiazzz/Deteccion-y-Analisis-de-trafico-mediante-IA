# Bitácora de avance

Se actualiza al cerrar cada sesión de trabajo. El registro va con lo más reciente arriba.
Plan de referencia: [`Plan_de_Accion_VideoTracking.md`](Plan_de_Accion_VideoTracking.md).

## Estado por fase

| Fase | Estado | Nota |
|---|---|---|
| F0 Preparación | En curso | Falta: Tailscale, GitHub, Drive, roles, umbral F1, Anexo A |
| F1 Pipeline mínimo | **Hecha** | |
| F2 Datos propios | En curso | **Herramientas listas** (anonimizar, partición, pre-anotación, lector de CVAT). Falta: decisiones, captura y anotación |
| F3 Evaluación base | **F3a hecha**; F3b/c listas para correr | Evaluación con CVAT, calibración y regla del jinete (E2) implementadas. Esperan anotaciones |
| F4 Ajuste fino YOLO | **Código hecho y ensayado** | Ensayo en VisDrone en la GPU: +62 % de mAP50 en secuencias no vistas. Falta: entrenar con sus datos |
| F5 Tracking y analítica | **Hecha** (salvo error de conteo) | Calibración del tracker hecha para los 4 modelos en VisDrone |
| F6 Reportes y tablero | **Hecha** (local) | Tablero con fondo animado (upstream-fx) y tema oscuro |
| F7 Cierre | Pendiente | |

## Pendientes del equipo

- [x] Crear el usuario administrador de CVAT (`camilo`)
- [ ] Instalar Tailscale en los 3 PCs y configurar `CVAT_HOST` (pasos en el README)
- [ ] Probar que los 3 entran a CVAT desde sus casas (hito de F0)
- [ ] Crear el repo en GitHub y conectar el remoto (`git remote add origin <url>` + `git push -u origin main`)
- [x] Descargar VisDrone2019-MOT val
- [ ] Crear la carpeta compartida en Drive (solo material anonimizado)
- [ ] Asignar roles I1, I2 e I3
- [ ] Decidir el umbral mínimo de F1 por clase: 0,60 o 0,65
- [ ] Actualizar la propuesta con el Anexo A del plan
- [ ] PyCharm: volver a seleccionar el intérprete `.venv` (ahora es Python 3.12)
- [ ] Crear el proyecto en CVAT con las etiquetas del README (incluye `ignore`)
- [ ] Revisar y commitear el trabajo desde el primer commit (todo sin commit, a pedido)
- [x] Decidir `--track-thresh` por defecto (0,3 YOLO26n), agregar YOLO26s, dedup como experimento aparte
- [ ] Probar el tablero: `.\.venv-rocm\Scripts\python -m streamlit run app\dashboard.py`
- [ ] Decidir tracker y modelo final cuando haya datos propios (ver `Explicacion_Fase5_Fase6.md` §6)
- [ ] Fase 2: resolver los puntos [ABIERTO] de `Guia_Anotacion.md` y `Protocolo_Captura.md`
- [ ] Con el primer clip real: anonimizar, revisar `*_check.jpg` (¿se escapa alguna cara o placa?) y exportar el primer zip de CVAT para probar `evaluate --cvat`
- [ ] Decidir los umbrales por defecto por modelo (en VisDrone: YOLO26s 0,3, RT-DETR 0,4, Faster R-CNN ≥ 0,7) o esperar a calibrar con sus datos

## Registro

### 2026-11-11 — Ajuste fino de YOLO26n con MIO-TCD (Colab)

Informe completo: [`Experimento_MIO_TCD.md`](Experimento_MIO_TCD.md).

- Conversor de MIO-TCD (oficial y exportación de Roboflow), `train --mio-tcd`, cuaderno `notebooks/mio_tcd_colab.ipynb`.
- La copia de Roboflow `thesis-ltv3y/mio_tcd-f3kk4` se descartó: una sola caja por imagen (mAP50 0,410).
- MIO-TCD oficial, 20.000 imágenes, 30 épocas en T4: mAP50 0,866 y mAP50-95 0,632 en su validación. Vehículos con mAP50 de 0,91 a 0,97; peatones 0,526 (recall 0,45).
- El modelo ajustado ahora usa los nombres de COCO (`person`), así que las 6 clases parten de los pesos preentrenados. No mejoró a los peatones.
- Pendiente: comparar `best.pt` con `yolo26n.pt` en la prueba fija propia.

### 2026-10-07 (noche) — Herramientas de F2 a F4 y fondo del tablero

Explicación completa: [`Explicacion_Fase2_Fase4_Herramientas.md`](Explicacion_Fase2_Fase4_Herramientas.md). **Sin commits**, a pedido.

**Hecho**
- Tablero: fondo animado upstream-fx (MIT), revisado, incluido en `app/static/` y en tema oscuro; verificado en Edge con capturas automáticas.
- `anonymize`:
  - difumina cabezas, placas y parabrisas de vehículos cercanos, y no guarda el audio;
  - genera una hoja de control `*_check.jpg` por video.
- `split`: prueba fija + 5 grupos, sin partir sesiones, estratificado por punto y franja.
- `prelabel`: pre-anotaciones MOT 1.1 para CVAT, con YOLO26x y jinetes unidos.
- `evaluate --cvat`: lee las exportaciones de CVAT. Además: `--sequences` y `--riders`.
- `calibrate`: barrido del umbral del tracker por modelo.
- `train`: exporta el dataset a formato YOLO y hace el ajuste fino (las regiones ignoradas se pintan de gris).
- Video escrito con ffmpeg/x264 (`imageio-ffmpeg`): archivos ~20 veces más livianos que con el códec de Windows.
- 41 pruebas; las capas respetan la regla de dependencias.

**Resultados**
- Anonimización: la primera versión dejaba ver al conductor por el parabrisas; corregido con la franja de cabina para vehículos ≥ 250 px de alto.
- Calibración (VisDrone, mejor HOTA): YOLO26n 0,2–0,3; YOLO26s 0,3 (0,363 contra 0,307 con 0,6); RT-DETR-L 0,4 (0,381 contra 0,344); Faster R-CNN ≥ 0,7 (0,352). Cada modelo pide su propio umbral: YOLO bajo, Faster R-CNN alto.
- Ensayo de ajuste fino (YOLO26n, 5 secuencias de entrenamiento, 2 no vistas, 12,5 min de GPU): mAP50 0,163 → 0,264, F1 macro 0,159 → 0,293, HOTA 0,306 → 0,401. **Bus y camión quedan en 0** porque no había ejemplos al entrenar: hay que asegurarlos en cada grupo de la partición.

**Hallazgos**
- ROCm 7.2.1 en Windows no compila el kernel de entrenamiento de normalización por lotes de MIOpen. Se entrena con MIOpen apagado (automático); es más lento, pero funciona.
- El códec H.264 de Windows (Media Foundation) ignora el ajuste de calidad; por eso se cambió a ffmpeg.

**Siguiente**
- Todo lo que queda necesita sus datos o decisiones (ver pendientes).

### 2026-10-07 (tarde) — Fase 5 (analítica) y Fase 6 (reportes y tablero)

Explicación completa: [`Explicacion_Fase5_Fase6.md`](Explicacion_Fase5_Fase6.md). **Sin commits**, a pedido.

**Decisiones del equipo**
- `--track-thresh` por defecto: 0,3 para YOLO26n y 0,6 para los demás.
- Agregar YOLO26s a la comparación.
- `--dedup` como experimento aparte.
- Sin commits.

**Hecho**
- F5:
  - zonas con permanencia;
  - 5 alertas (contraflujo, horario restringido, clase no permitida en zona, permanencia prolongada, aglomeración);
  - hora real desde el nombre del archivo o `--start`;
  - configuración por punto con líneas, zonas y restricciones;
  - video anotado con zonas y alertas, en H.264;
  - tracker elegible (`--tracker bytetrack|botsort`).
- F6:
  - resumen agregado (composición, intervalos, pico, permanencia);
  - CSV agregados;
  - reporte PDF de 2 páginas;
  - tablero Streamlit local;
  - `vtia/compose.py` compartido por consola y tablero.
- Ejemplos: `sample_vehicles.yaml` (dos calzadas, sentidos permitidos, camiones restringidos) y `sample_people.yaml` (plaza con zonas simuladas). Se agregó el video `people-walking.mp4`.
- 29 pruebas; las capas respetan la regla de dependencias.
- `.streamlit/config.toml`: el tablero solo escucha en el propio PC (no en la red) y no envía estadísticas de uso, por la Ley 1581.

**Resultados**
- Autopista: 9 cruces sin contraflujo; con `--start 07:00`, 1 alerta de horario restringido (camión, 07:00:16).
- Plaza: 19 cruces, 11 visitas y 8 alertas.
- VisDrone:
  - YOLO26s duplica el F1 macro de YOLO26n (0,262 contra 0,136) e iguala el mAP de Faster R-CNN, a 37 fps.
  - E4: BoT-SORT reduce a la mitad los cambios de ID de Faster R-CNN (1.487 → 762) pero empeora a YOLO y RT-DETR. ByteTrack queda por defecto.

**Siguiente**
- Probar el tablero.
- F2 (decisiones de captura y anotación) para destrabar el error de conteo, F3b y F4.

### 2026-10-07 (tarde) — Video de salida anotado

**Hecho**
- `count --save-video` genera `outputs/<video>/annotated.mp4`: cajas coloreadas por clase con su ID, el punto que se cuenta, la línea virtual (se pone verde en cada cruce) y un panel con los conteos acumulados. Videos 4K se guardan en 1080p.
- El caso de uso `process_video` acepta un observador por cuadro (`on_frame`); el video lo implementa como adaptador (`adapters/annotated_video.py`), sin guardar nada en memoria.
- Video de ejemplo con `--track-thresh 0.3`: 9 cruces, 23,9 fps en la GPU incluyendo escribir el video (29 MB).

**Siguiente**
- Igual que la entrada anterior (decisiones pendientes de `Explicacion_Fase3a.md` §8).

### 2026-10-07 (noche) — GPU AMD, Fase 3a y borradores de Fase 2

Explicación completa: [`Explicacion_Fase3a.md`](Explicacion_Fase3a.md). **Sin commits**, a pedido.

**Hecho**
- GPU AMD RX 9060 XT funcionando con PyTorch 2.9.1 + ROCm 7.2.1 en `.venv-rocm`, sin tocar drivers ni el `.venv` principal.
- Arquitectura: detección y tracking separados, con el mismo ByteTrack (paquete `trackers`) para todos los modelos; puertos `Detector`/`Tracker`; caché de detecciones.
- Detectores: YOLO26n, RT-DETR-L y Faster R-CNN R50-FPN v2, todos a 640 px de lado mayor.
- Módulo de evaluación (F3a): mAP (pycocotools), matriz de confusión con fondo, P/R/F1, HOTA/MOTA/MOTP/IDF1/cambios de ID (TrackEval con parche para numpy 2), regiones ignoradas.
- Validación: 15 pruebas (una reproduce la tabla 10.1 de la propuesta) y oráculo en VisDrone = 1,000 en todas las métricas.
- Opciones nuevas: `--dedup IOU` (deduplicación de cualquier clase) y `--track-thresh T` (umbral de ByteTrack).
- Borradores: `Guia_Anotacion.md` y `Protocolo_Captura.md`. Etiqueta `ignore` agregada a CVAT en el README.

**Resultados en VisDrone (sin entrenar, desde dron; no representa la cámara del proyecto)**

| Modelo | mAP50 | F1 macro | HOTA | MOTA | IDF1 | fps GPU |
|---|---|---|---|---|---|---|
| YOLO26n (2,6 M) | 0,168 | 0,136 | 0,196 | 0,076 | 0,138 | 65,5 |
| RT-DETR-L (33 M) | 0,300 | 0,270 | 0,344 | 0,199 | 0,355 | 22,5 |
| Faster R-CNN (43,7 M) | 0,248 | 0,252 | 0,346 | 0,204 | 0,418 | 12,9 |

**Hallazgos**
- La validación encontró dos errores, ya corregidos: el parche de numpy pisaba `np.bool`, y las regiones ignoradas no se aplicaban a las anotaciones.
- Cajas duplicadas de distinta clase: 7 % (YOLO26n), 53 % (RT-DETR) y 27 % (Faster R-CNN) de las detecciones. `--dedup 0.7` mejora el F1 y el tracking pero baja el mAP. No explica los 1.487 cambios de ID de Faster R-CNN; esa causa sigue abierta.
- Cada modelo necesita su propio umbral de tracker. Con `--track-thresh 0.3`, YOLO26n mejora mucho (MOTA 0,076 → 0,177) y el video de ejemplo da **9/9**; RT-DETR y Faster R-CNN empeoran.
- Se confirman las confusiones previstas en la propuesta: bicicleta↔moto y bus→camión.
- La comparación mezcla arquitectura y tamaño (2,6 M contra 33–44 M de parámetros).

**Siguiente**
- Decisiones de la sección 8 de `Explicacion_Fase3a.md`.
- F2: con ciudad, puntos y celulares definidos, cerrar los borradores y empezar la captura.
- F3b: elegir umbrales por modelo sobre validación (lo pedirá la partición de los datos propios).

### 2026-10-07 — Fase 0 (parte técnica) y Fase 1

**Hecho**
- Entorno: `.venv` recreado con Python 3.12.10; `requirements.txt` fijado (ultralytics 8.4.174, lap 0.5.13, pyyaml 6.0.3).
- Git local inicializado con `.gitignore`: datos, videos, resultados y pesos nunca van a git.
- CVAT v2.78.0 en `C:\Users\USER\cvat`, corriendo en http://localhost:8080. Consume unos 2,9 GB de RAM.
- Video de ejemplo descargado en `data/samples/vehicles.mp4`: autopista en 4K, 25 fps, 538 cuadros.
- Pipeline F1 con arquitectura limpia (`vtia/domain`, `application`, `adapters`, `cli.py`) y pruebas de la lógica de conteo (`tests/test_counting.py`).
- Plan actualizado: arquitectura (§1.4) y decisiones (§0).

**Decisiones**
- Arquitectura limpia pragmática; nombres en inglés en el código.
- Modelo principal: YOLO26n.
- Un track cuenta como máximo una vez por línea, para que el temblor sobre la línea no duplique conteos.
- La clase de un cruce es la clase mayoritaria del track, no la del cuadro del cruce.

**Resultados (video de ejemplo, CPU Ryzen 5 9600X)**

| Modelo | Cruces | Velocidad | Observación |
|---|---|---|---|
| YOLO26n | 8 | 29,1 fps | Pierde 1: duplica el carro negro y el tracker le pasa ese ID al carro blanco, que ya no se cuenta |
| YOLO11n | 10 | 28,1 fps | Cuenta 2 veces el carro negro (dos tracks cruzan en el mismo cuadro) |

Conteo real estimado: 9, por revisión visual de los cuadros de cada evento; el video no se revisó completo. El camión del inicio no cuenta porque arranca ya pasada la línea.

**Hallazgos**
- **TrackEval** (último commit de 2022) usa `np.float`, `np.int` y `np.bool`, que numpy 2 eliminó (81 usos). Hay que decidirlo en F3a: parche, fork mantenido o `py-motmetrics`.
- **VisDrone desde Google Drive:** `gdown` falla porque el archivo superó la cuota de descargas. Hay que bajarlo desde el navegador.
- Las detecciones duplicadas de un mismo vehículo son la principal fuente de error de conteo observada. Se ataca en F3 y F5, con datos de referencia: umbral de confianza, BoT-SORT, supresión de duplicados.

**Siguiente**
- Equipo: cerrar los pendientes de F0.
- F3a: módulo de evaluación con VisDrone, cuando esté descargado.
- F2: protocolo de captura y guía de anotación, cuando definan ciudad, puntos y celulares.

### 2026-11-10 — ETAPA 1 (infraestructura de evaluación científica, sin datos propios)

**Implementado en esta entrega**
- Evaluador de conteo manual vs. automático: diferencia firmada/absoluta, error porcentual (N/D si el conteo manual es cero), MAE, RMSE y desgloses por clase, video, punto, franja horaria, dirección y línea.
- `validate-dataset`: inventario de videos y exportaciones CVAT, pendientes, huérfanos, duplicados, objetos por clase/secuencia/punto/franja, regiones ignore y secuencias inválidas.
- CVAT MOT 1.1: validación estricta de columnas, valores finitos, IDs, clases, marcas, visibilidad, cajas, duplicados frame/track y rango de cuadros. Los errores ya no se descartan silenciosamente.
- Particiones: validación de manifiesto, cobertura, cinco folds, aislamiento de sesiones y etiquetas permitidas. Se genera un manifiesto JSON con semilla, proporción, folds y SHA-256 del CSV.
- Evaluación propia: requiere manifiesto y partición explícita. `evaluate` permite seleccionar partición; `calibrate` rechaza el conjunto `test`; entrenamiento continúa excluyendo `test`.
- Caché de detecciones: clave basada en hash del video, identidad/configuración del detector y huella de pesos cuando está disponible; escrituras atómicas y validación del hash del archivo cacheado.
- Reporte de evaluación: exportación de resumen JSON y PDF además de CSV/Markdown.
- Guía: `Docs/Guia_Evaluacion_Cientifica.md`; README actualizado con el nuevo flujo.

**Pruebas nuevas**
- `tests/test_scientific_validation.py`: 19 pruebas para conteo, particiones, MOT 1.1, robustez, métricas por secuencia, PDF y caché.
- `tests/test_cli_scientific.py`: 5 pruebas para protección de particiones y disponibilidad del CLI sin dependencias de inferencia.

**Limitaciones pendientes**
- No hay videos propios ni anotaciones CVAT reales dentro del ZIP: el inventario y los resultados propios quedan pendientes de datos.
- La evaluación de calidad semántica de las etiquetas requiere revisión humana; la validación estructural no la sustituye.
- La estratificación temporal y por punto se informa; no se garantiza todavía equilibrio de clases en cada fold cuando el volumen por clase es insuficiente.
- El análisis de robustez agrupa métricas según condiciones registradas manualmente; no clasifica automáticamente iluminación/oclusiones desde los píxeles. Tampoco se mide memoria RAM/GPU ni se consolida todavía la media ± desviación estándar de cinco ejecuciones de validación cruzada.
- La suite completa sigue dependiendo del entorno Python 3.12 y de las dependencias de evaluación (incluido `supervision`); debe verificarse allí antes de declarar la fase cerrada.

**Ampliación de ETAPA 1 (misma entrega)**
- `evaluate` conserva métricas individuales por secuencia, necesarias para análisis por condición, y exporta `per_sequence.csv` por modelo.
- `--conditions` permite asociar condiciones de captura anotadas manualmente (iluminación, oclusión, densidad, tamaño y distancia) con métricas por secuencia. Menos de tres secuencias por condición se informa como `insufficient_evidence`.
- Se exporta PDF también para evaluación de conteo.
- El resumen distingue FPS de inferencia y FPS de procesamiento integral; incluye parámetros y tamaño del checkpoint cuando se pueden medir. Memoria RAM/GPU queda N/D/no medida en esta entrega.
- Se corrigieron las franjas del parser de nombres para corresponder a las ventanas propuestas del protocolo: 06:30–09:00, 11:30–14:00 y 16:30–18:30. Se rechazan fechas/horas inválidas.
- `accuracy` se conserva como alias por compatibilidad, pero el resumen científico ahora lo denomina `micro_recall`, ya que no representa accuracy convencional con verdaderos negativos.
- El CLI ahora carga dependencias pesadas de modelos/evaluación solo al ejecutar esos comandos; `--help`, validación de datos y evaluación de conteos pueden ejecutarse sin importar trackers/modelos.
