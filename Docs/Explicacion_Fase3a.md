# Qué hicimos: GPU, Fase 3a y borradores de la Fase 2

Fecha: 2026-10-07 (sesión nocturna). La bitácora con el estado al día es [`AVANCE.md`](AVANCE.md).

---

## 1. Resumen

- **Tu GPU AMD funciona para IA.** PyTorch reconoce la RX 9060 XT. YOLO26n corre a 65 fps, RT-DETR-L a 23 y Faster R-CNN a 13. Los tres modelos pueden correr en tu PC; Colab queda de respaldo.
- **El sistema ahora tiene los 3 detectores del plan,** todos con el mismo tracker, para que la comparación sea justa.
- **El módulo de evaluación está terminado y validado** (hito de la Fase 3a). Calcula mAP, la matriz de confusión con fila de fondo, precisión/recall/F1 por clase, HOTA, MOTA, MOTP, IDF1 y cambios de ID.
- **Primera comparación de los 3 modelos en VisDrone,** más dos experimentos que explican el error de conteo del video de ejemplo. Ahora ese video da **9 de 9** con un ajuste.
- **Borradores de la guía de anotación y del protocolo de captura,** con los puntos que tienen que decidir ustedes marcados.
- Nada se commiteó, como pediste.

---

## 2. GPU AMD

| | |
|---|---|
| Entorno | `.venv-rocm` (aparte del `.venv` normal; no se tocaron drivers) |
| PyTorch | 2.9.1 + ROCm 7.2.1, la versión oficial de AMD para Windows |
| Uso | Igual que siempre, cambiando `python` por `.venv-rocm\Scripts\python` |
| Cómo recrearlo | README, sección "GPU AMD" |

Velocidad medida en las 2.846 imágenes de VisDrone (incluye leer cada imagen del disco):

| Modelo | Parámetros | GPU (fps) |
|---|---|---|
| YOLO26n | 2,6 M | 65,5 |
| RT-DETR-L | 33,0 M | 22,5 |
| Faster R-CNN R50-FPN v2 | 43,7 M | 12,9 |

---

## 3. Cambios en la arquitectura

**Antes:** YOLO traía su propio tracker interno.
**Ahora:** detectar y seguir son dos pasos separados, y el tracker es el mismo para todos los modelos.

```
video ─► Detector (YOLO | RT-DETR | Faster R-CNN) ─► [deduplicar, opcional] ─► ByteTrack ─► conteo / evaluación
```

**Por qué:**
- **Comparación justa:** si cada modelo trajera su propio tracker, no sabríamos si la diferencia viene del detector o del tracker.
- **Plan §1.2:** las detecciones se guardan una vez (`CachedDetector`) y todo lo demás las reutiliza. Re-evaluar con otro ajuste toma minutos, sin volver a correr los modelos.

**Archivos nuevos por capa:**

| Capa | Archivo | Qué hace |
|---|---|---|
| Dominio | `evaluation.py` | IoU, regiones ignoradas, emparejamiento, matriz de confusión, P/R/F1 |
| Dominio | `postprocess.py` | Elimina cajas duplicadas de cualquier clase (opcional) |
| Aplicación | `ports.py` | Interfaces `Detector` y `Tracker` |
| Aplicación | `evaluate.py` | Caso de uso: evaluar contra un dataset anotado |
| Aplicación | `dedup.py` | Envuelve un detector y le quita duplicados |
| Adaptadores | `ultralytics_detector.py` | YOLO y RT-DETR |
| Adaptadores | `faster_rcnn_detector.py` | Faster R-CNN de torchvision, a la misma resolución que los otros (lado mayor 640 px) |
| Adaptadores | `bytetrack_tracker.py` | ByteTrack del paquete `trackers` |
| Adaptadores | `cached_detector.py` | Guarda y reutiliza detecciones |
| Adaptadores | `visdrone.py` | Lector de VisDrone2019-MOT |
| Adaptadores | `coco_map.py` | mAP con pycocotools |
| Adaptadores | `trackeval_metrics.py` | Métricas de tracking con TrackEval |

**Dos decisiones técnicas que tomé por el camino:**
- **ByteTrack del paquete `trackers`, no de `supervision`:** el de `supervision` está obsoleto y desaparece en su próxima versión. `trackers` es el reemplazo oficial, del mismo equipo, y trae BoT-SORT con la misma interfaz, lo que deja listo el experimento E4.
- **Parche de TrackEval:** restaura solo los alias que numpy 2 quitó (`np.float`, `np.int`). Mi primera versión también pisaba `np.bool`, que numpy 2 volvió a usar, y rompía numpy por dentro. Lo detectó la validación de la sección 4.

---

## 4. Cómo sé que la evaluación calcula bien

1. **Recalcula la tabla de la propuesta.** Con la matriz ilustrativa de la sección 10, el código da exactamente los mismos valores que la tabla 10.1 (precisión 0,912 de peatón, F1 macro 0,782, etc.).
2. **"Oráculo" sobre VisDrone real:** si se le pasan las anotaciones como si fueran predicciones, todo debe dar 1,0. **Al principio no dio 1,0,** y así aparecieron dos errores:
   - el del parche de numpy (sección 3);
   - **las regiones ignoradas se aplicaban a las detecciones pero no a las anotaciones.** Por eso 2.782 personas sentadas en triciclos (clase que ignoramos) contaban como "no detectadas".

   Corregidos los dos, el oráculo da **1,000 en mAP, HOTA, MOTA, MOTP, IDF1 y F1, sin confusiones**.
3. **15 pruebas automáticas.** Incluyen casos de un objeto perdido (MOTA = 1 − 1/12), una caja de clase equivocada (cuenta como confusión, no como pérdida), regiones ignoradas y el adaptador de ByteTrack.

**Reglas de la evaluación** (vale la pena leerlas antes de la sustentación):
- El emparejamiento ignora la clase: la detección más confiable se queda con su objeto de mayor IoU (≥ 0,5). Así un bus detectado como camión cuenta como *confusión bus→camión*, no como un fallo más un falso positivo.
- **Fila "background":** detecciones que no corresponden a nada (falsos positivos). **Columna "missed":** objetos que nadie detectó.
- Lo que cae mayoritariamente dentro de una región ignorada no cuenta, ni de un lado ni del otro.
- La matriz de confusión usa confianza ≥ 0,25; el mAP usa todas las detecciones.
- El tracking se evalúa sin distinguir clases: importa que el ID se mantenga.

---

## 5. Resultados en VisDrone2019-MOT (validación)

**Advertencia antes de leer los números:** VisDrone es video **desde dron**, con objetos diminutos vistos desde arriba, y los modelos están **sin entrenar** (pesos COCO) a 640 px. Es muy distinto de la cámara a 3–6 m del proyecto. **Estos números no anticipan el desempeño con sus datos.** El objetivo era validar el módulo y los adaptadores, y sacar lecciones de método.

| Modelo | mAP50 | mAP50-95 | AP pequeños | F1 macro | HOTA | MOTA | IDF1 | Cambios de ID | fps |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26n | 0,168 | 0,078 | 0,017 | 0,136 | 0,196 | 0,076 | 0,138 | 43 | 65,5 |
| RT-DETR-L | **0,300** | **0,148** | **0,054** | **0,270** | 0,344 | 0,199 | 0,355 | 116 | 22,5 |
| Faster R-CNN | 0,248 | 0,109 | 0,035 | 0,252 | **0,346** | **0,204** | **0,418** | 1.487 | 12,9 |

Detalle por clase y matrices de confusión: `outputs/eval/visdrone/summary.md`.

**Lo que se ve:**
- **Aparecen las confusiones que anticipaba la propuesta.** Bicicleta→moto: RT-DETR clasifica 2.034 bicicletas como motos. Bus→camión: 177 de los 264 buses. Peatón y ciclista también se confunden. La sección 10.2 de la propuesta acertó en los modos de falla.
- **El bus es la clase más débil:** casi ningún modelo los reconoce desde arriba. Con solo 264 casos, coincide con el riesgo de la propuesta de tener "pocas instancias de bus/camión".
- **"Camión" tiene precisión muy baja:** COCO llama `truck` a furgonetas y pickups, y en VisDrone las furgonetas están anotadas como carro (`van` → `car`). Es el mismo desajuste de COCO anotado en el plan.
- **YOLO26n es el más rápido pero el más débil aquí,** y **la comparación está desbalanceada por tamaño:** 2,6 M de parámetros contra 33 M y 43,7 M. Hoy la tabla mide arquitectura *y* tamaño a la vez (decisión pendiente, sección 8).

---

## 6. Dos experimentos: duplicados y umbral del tracker

### 6.1 Cajas duplicadas

El error del video de ejemplo empezaba con un carro detectado dos veces. Medí cuántas detecciones (confianza ≥ 0,1) tienen un "gemelo" (IoU > 0,7) en el mismo cuadro:

| Modelo | Con gemelo | Gemelo de otra clase |
|---|---|---|
| YOLO26n | 7 % | 7 % |
| RT-DETR-L | 57 % | 53 % |
| Faster R-CNN | 27 % | 27 % |

Casi siempre es la misma caja reportada con dos clases (carro y camión). Agregué la opción `--dedup 0.7`, que se queda con la más confiable sin importar la clase. Viene **apagada por defecto**.

| Con `--dedup 0.7` | mAP50 | F1 macro | HOTA | MOTA | IDF1 | Cambios de ID |
|---|---|---|---|---|---|---|
| RT-DETR-L | 0,300 → 0,269 | 0,270 → **0,307** | 0,344 → 0,349 | 0,199 → 0,207 | 0,355 → 0,361 | 116 → 114 |
| Faster R-CNN | 0,248 → 0,223 | 0,252 → **0,269** | 0,346 → 0,358 | 0,204 → **0,245** | 0,418 → 0,430 | 1.487 → 1.331 |

Mejora la clasificación y el tracking, pero baja el mAP. Y **no explica** los cambios de ID de Faster R-CNN: solo los reduce un 10 %. Esa causa sigue abierta.

### 6.2 Umbral del tracker (`--track-thresh`)

ByteTrack solo crea un track nuevo si la detección supera cierto umbral de confianza. Con el valor por defecto de la librería (≥ 0,7), en el video de ejemplo el carro blanco recién obtenía track **después** de cruzar la línea: su confianza rondaba 0,44–0,64. Agregué la perilla `--track-thresh` (por defecto 0,6, que reproduce el comportamiento de la librería).

| `--track-thresh` | Video de ejemplo | VisDrone YOLO26n (HOTA / MOTA / IDF1) | VisDrone RT-DETR-L (MOTA) | VisDrone Faster R-CNN (MOTA) |
|---|---|---|---|---|
| 0,6 (por defecto) | 8 de 9 | 0,196 / 0,076 / 0,138 | 0,199 | 0,204 |
| 0,3 | **9 de 9** | **0,286 / 0,177 / 0,295** | −0,121 | 0,071 |

**Lección:** cada modelo "habla" con confianzas en escalas distintas. YOLO26n da confianzas bajas y necesita un umbral bajo. RT-DETR y Faster R-CNN dan muchos falsos positivos con confianza media, y un umbral bajo los convierte en tracks falsos. **No existe un umbral bueno para todos:** hay que elegirlo por modelo sobre datos de validación, como dice la sección 2.2 del plan. Ahora hay evidencia de por qué.

No cambié ningún valor por defecto: la decisión es de ustedes (sección 8).

---

## 7. Borradores de la Fase 2

- [`Guia_Anotacion.md`](Guia_Anotacion.md): clases con casos frontera, cómo dibujar cajas, regiones `ignore`, reglas del track, control de calidad y checklist.
- [`Protocolo_Captura.md`](Protocolo_Captura.md): qué grabar, criterios de puntos, configuración del celular, franjas, nombres de archivo, planilla, procedimiento en sitio y privacidad.

**Una propuesta importante del protocolo:** grabar sesiones largas y **recortar de ellas** los clips cortos, en vez de grabarlos aparte. Una salida a campo sirve para todo. Obliga a que la partición se agrupe **por sesión**: si no, las métricas salen infladas.

Agregué la etiqueta `ignore` a la lista de etiquetas de CVAT en el README, porque la evaluación ya la usa.

---

## 8. Decisiones que necesito de ustedes

1. **Umbral del tracker:** ¿fijar `--track-thresh 0.3` como valor por defecto para YOLO26n en el conteo, o esperar a calibrarlo con sus datos?
2. **Comparación por tamaño:** ¿agregar YOLO26s o YOLO26m (más grandes), para separar "arquitectura" de "tamaño"? Cuesta minutos en tu GPU.
3. **Deduplicación:** ¿la dejamos como opción de experimento o la incluimos en la comparación oficial (E2)?
4. **Fase 2:** ciudad, puntos, celulares y los casos frontera de la guía (motocarros, chivas, furgonetas, etc.).
5. **Commits:** todo lo de esta sesión está sin commitear, para que lo revises.
