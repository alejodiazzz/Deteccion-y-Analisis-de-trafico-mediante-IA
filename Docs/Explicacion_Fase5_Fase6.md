# Qué hicimos: Fase 5 (analítica) y Fase 6 (reportes y tablero)

Fecha: 2026-10-07. La bitácora con el estado al día es [`AVANCE.md`](AVANCE.md).

---

## 1. Resumen

- **El sistema ya hace todo lo que pide la propuesta** sobre un video:
  - conteo por línea y sentido;
  - zonas con tiempo de permanencia;
  - cinco tipos de alerta;
  - distribución temporal con hora pico;
  - reporte PDF, CSV y video anotado.
- **Hay un tablero web local:** se elige o sube un video, se presiona *Analizar* y se ve todo, con descargas.
- **Comparación ampliada en VisDrone:** YOLO26s se suma a los modelos, y el experimento E4 compara ByteTrack con BoT-SORT.
- **Lo que sigue pendiente necesita sus datos:** el error de conteo en videos largos, la calibración de umbrales y la publicación del tablero.
- **29 pruebas automáticas**, todas pasan. Nada está commiteado, como pediste.

---

## 2. Cómo usarlo

**Tablero:**
```powershell
cd C:\Users\USER\Documents\TrabajoDeCampo
.\.venv-rocm\Scripts\python -m streamlit run app\dashboard.py
```
Se abre en el navegador, en http://localhost:8501.

**Consola:**
```powershell
.\.venv-rocm\Scripts\python -m vtia.cli count data\samples\vehicles.mp4 --point configs\points\sample_vehicles.yaml --start "2026-10-07 07:00" --save-video
```
Deja en `outputs\vehicles\`: `report.pdf`, `annotated.mp4` y 8 CSV. La lista completa está en el README.

---

## 3. Fase 5: analítica

### 3.1 Configuración del punto de cámara

Todo se configura por punto en un YAML (`configs/points/`), sin tocar código:

```yaml
lines:                       # líneas de conteo
  - name: left_lanes
    start: [850, 1300]
    end: [1700, 1300]
    to_right: toward_camera
    to_left: away_from_camera
    allowed: toward_camera   # opcional: el otro sentido = contraflujo
zones:                       # zonas de interés con reglas opcionales
  - name: left_carriageway
    polygon: [[1380, 800], [1660, 800], [1740, 2160], [180, 2160]]
    forbidden: [pedestrian, bicycle]
    max_dwell_s: 120
    max_occupancy: 8
restrictions:                # restricciones horarias por clase
  - class: truck
    hours: ["06:00-08:00", "17:00-19:00"]
```

El lector valida el archivo y da errores claros. Por ejemplo: un `allowed` que no es uno de los dos sentidos, nombres repetidos o un polígono de menos de 3 puntos.

### 3.2 Zonas y permanencia (`domain/zones.py`)

- Un objeto está "dentro" de una zona cuando su punto de apoyo (la base de la caja) cae dentro del polígono.
- Una visita sigue abierta mientras el objeto se vea dentro, y se cierra tras **1 segundo sin verlo**. Así, si el tracker lo pierde un instante, la visita no se parte en dos.
- La clase de cada visita es la mayoritaria del objeto, igual que en el conteo.

### 3.3 Alertas (`domain/alerts.py`)

| Alerta | Cuándo salta | Configuración |
|---|---|---|
| Contraflujo | Un objeto cruza una línea en el sentido no permitido | `allowed` en la línea |
| Horario restringido | Una clase restringida cruza una línea dentro de un horario prohibido | `restrictions` (requiere la hora real) |
| Clase no permitida en zona | Una clase prohibida pasa ≥ 0,5 s dentro de una zona (p. ej. peatón en calzada) | `forbidden` en la zona |
| Permanencia prolongada | Un objeto supera `max_dwell_s` dentro de una zona | `max_dwell_s` |
| Aglomeración | La zona llega a `max_occupancy` objetos a la vez | `max_occupancy` |

Cada alerta salta **una sola vez por objeto**. La de aglomeración vuelve a saltar solo cuando la zona se vació y se llenó de nuevo. Las ventanas horarias pueden cruzar la medianoche (22:00-05:00).

### 3.4 Hora real (`domain/timeline.py`)

Sale del nombre del archivo que fija el protocolo de captura:
- `P1_20261015_0705.mp4` es una sesión que empezó a las 07:05;
- `P1_20261015_0705_s0312.mp4` es un clip recortado 312 s después de ese inicio.

Si el nombre no sigue el formato, se usa `--start` o el campo del tablero. Si no hay hora, todo funciona igual, excepto la regla de horario restringido, y los reportes muestran tiempo transcurrido.

### 3.5 Video anotado

Ahora también dibuja las zonas (celeste), pinta de rojo los objetos con alerta y lista las últimas alertas en el panel. Se guarda en **H.264** usando el códec que ya trae Windows, así que se ve en el navegador (el tablero) y en cualquier reproductor. En Linux o Colab usa otro formato que abre VLC.

### 3.6 Experimento E4 y YOLO26s (VisDrone)

| Modelo | Tracker | HOTA | MOTA | IDF1 | Cambios de ID | mAP50 | F1 macro |
|---|---|---|---|---|---|---|---|
| YOLO26n (umbral 0,3) | ByteTrack | 0,286 | 0,177 | 0,295 | 294 | 0,168 | 0,136 |
| YOLO26n (umbral 0,3) | BoT-SORT | 0,252 | 0,154 | 0,235 | 155 | | |
| **YOLO26s** | ByteTrack | 0,307 | 0,182 | 0,297 | 91 | **0,248** | **0,262** |
| YOLO26s | BoT-SORT | 0,284 | 0,159 | 0,261 | 59 | | |
| RT-DETR-L | ByteTrack | 0,344 | 0,199 | 0,355 | 116 | 0,300 | 0,270 |
| RT-DETR-L | BoT-SORT | 0,323 | 0,202 | 0,319 | 117 | | |
| Faster R-CNN | ByteTrack | 0,346 | 0,204 | 0,418 | 1.487 | 0,248 | 0,252 |
| Faster R-CNN | BoT-SORT | **0,357** | **0,274** | **0,430** | **762** | | |

(mAP y F1 dependen solo del detector, no del tracker.)

- **El tamaño importa:** YOLO26s (unos 10 M de parámetros) duplica el F1 macro de YOLO26n e iguala el mAP de Faster R-CNN, a 37 fps. Pasa a ser un candidato serio para el modelo final.
- **BoT-SORT no gana en general:** reduce a la mitad los cambios de ID de Faster R-CNN, pero empeora a los demás. Ojo: VisDrone es video desde dron en movimiento y le apagué a BoT-SORT la compensación de movimiento de cámara (pensada para cámaras fijas), así que en VisDrone queda en desventaja. **Hay que decidirlo con sus videos de cámara fija.** Mientras, ByteTrack queda por defecto.

---

## 4. Fase 6: reportes y tablero

### 4.1 El resumen (`application/summary.py`)

Una sola función arma, a partir del análisis, todo lo que muestran el PDF y el tablero:
- totales y composición por clase;
- conteos por línea y sentido;
- conteos por intervalo y el **intervalo pico**;
- permanencia por zona (visitas, promedio, máximo);
- alertas por tipo;
- datos de calidad.

El intervalo se elige solo: 10 s para clips, 1 min hasta media hora y 15 min para sesiones largas.

**Importante sobre la "hora pico":** con clips de 20 s solo se ve el pico *dentro del clip*. La hora pico real del punto sale de las sesiones largas (15–30 min) en distintas franjas, como propone el protocolo de captura.

### 4.2 El PDF (`adapters/pdf_report.py`)

Ocupa 2 páginas por video, con estas secciones:
1. **Encabezado:** video, punto, inicio, duración, modelo, tracker y equipo de proceso.
2. **Resumen:** cuatro cifras.
3. **Conteo por línea y sentido.**
4. **Gráfica de composición del tráfico.**
5. **Gráfica de distribución temporal,** con el pico resaltado.
6. **Permanencia en zonas.**
7. **Alertas:** por tipo y una lista con hora o segundo.
8. **Anexo de calidad:** objetos seguidos, confianza media y una advertencia de que los conteos deben validarse con conteos manuales.

Las gráficas usan un solo tono de azul, con el pico en un azul más oscuro y el valor escrito junto a cada barra. Todo el texto está en español; los nombres de los sentidos y las zonas salen del YAML, así que si los escriben en español, el PDF queda todo en español.

### 4.3 El tablero (`app/dashboard.py`)

- **Panel izquierdo:** subir o elegir un video, el punto de cámara, el detector, el tracker, un deslizador para el umbral y la hora de inicio (la detecta del nombre del archivo o se escribe a mano).
- **Barra de progreso** por cuadro mientras analiza.
- **Cuatro cifras arriba:** cruces, objetos seguidos, alertas y duración.
- **Pestañas:**
  - *Resumen:* gráficas interactivas con información al pasar el mouse, y la tabla por línea y sentido.
  - *Video:* el video anotado.
  - *Zonas y alertas:* tablas.
  - *Descargas:* PDF, CSV y video.

**Arquitectura:** el tablero es **otro punto de entrada** a la misma aplicación. Usa `vtia/compose.py`, el mismo módulo que la consola, así que no hay lógica duplicada y la consola y el tablero dan exactamente los mismos números. Lo comprobé con la prueba automática de Streamlit: 19 cruces, 58 objetos y 8 alertas en el video de la plaza, por los dos caminos.

Por ahora es **solo local**. Publicarlo es la Fase 8, y solo con videos anonimizados.

---

## 5. Videos de ejemplo y lo que muestran

| Video | Configuración | Resultado |
|---|---|---|
| Autopista (`vehicles.mp4`) | Una línea y una zona por calzada, sentido permitido, camiones restringidos 06:00-08:00 | 9 cruces en los sentidos correctos, sin contraflujos; con `--start 07:00`, **1 alerta de horario restringido** (el camión, 07:00:16) |
| Plaza (`people-walking.mp4`) | Una línea, una zona de encuentro (permanencia > 5 s, aglomeración ≥ 4) y una "calzada" simulada donde los peatones están prohibidos | 19 cruces y 8 alertas: 5 peatones en la calzada, 1 aglomeración y 2 permanencias |

Las zonas de la plaza son **simuladas**, para probar las reglas: la plaza no es una vía.

---

## 6. Pendientes y decisiones

**Necesitan sus datos:**
- **Error de conteo:** comparar el conteo automático con el conteo manual en las sesiones largas (meta de la propuesta: ≤ 15 %).
- **Calibrar por punto:** el umbral del tracker, las zonas y los límites de permanencia y aglomeración con video real de cada punto.
- **Decidir el tracker y el modelo final** (YOLO26n, YOLO26s, RT-DETR o Faster R-CNN) con sus métricas.

**Límites conocidos (anotados en el código):**
- La alerta de horario restringido se evalúa al cruzar una línea. Un camión que aparece pero no cruza ninguna línea no la dispara.
- Un objeto que cruza una línea y vuelve se cuenta una sola vez por línea. Es para que el temblor sobre la línea no duplique conteos.

**Siguientes fases posibles:**
- **F2:** con la ciudad, los puntos y los celulares definidos, cerrar los borradores y empezar la captura.
- **F4:** ajuste fino de YOLO. Necesita clips anotados.
- **F8:** publicar el tablero, clasificador en cascada, SAHI y clases locales.
