# Qué hicimos: Fase 0 y Fase 1

Fecha: 2026-10-07. Este documento explica lo construido y por qué. La bitácora con el estado al día es [`AVANCE.md`](AVANCE.md).

> **Nota posterior (misma noche, Fase 3a):** el pipeline evolucionó. El comando ahora es `python -m vtia.cli count ...`, el tracking pasó a ser un paso separado (detectar → ByteTrack del paquete `trackers`, igual para todos los modelos) y se agregaron Faster R-CNN y RT-DETR. Lo de abajo describe la primera versión; los cambios están en [`Explicacion_Fase3a.md`](Explicacion_Fase3a.md).

---

## 1. Resumen

- **Infraestructura lista en tu PC:** Python 3.12, git, CVAT corriendo, video de ejemplo descargado.
- **Primera versión del sistema funcionando de punta a punta:** recibe un video, detecta y sigue los vehículos y personas, cuenta cuántos cruzan una línea virtual en cada sentido y lo guarda en CSV.
- En el video de ejemplo cuenta 8 de unos 9 vehículos, a 29 cuadros por segundo en tu CPU, sin GPU.
- Queda pendiente lo que solo puede hacer el equipo: cuentas, Tailscale, GitHub y descargar VisDrone a mano.

---

## 2. Fase 0 — Preparación

| Qué | Dónde | Por qué |
|---|---|---|
| Entorno Python 3.12 | `.venv/` | Misma versión que Colab: lo que corre en tu PC corre igual allá. Reemplazó el 3.14, que estaba vacío |
| Dependencias fijadas | `requirements.txt` | Los 3 integrantes y Colab instalan exactamente las mismas versiones |
| Git local | `.gitignore` | Videos, datos, resultados y pesos nunca suben a git, por privacidad y tamaño |
| CVAT v2.78.0 | `C:\Users\USER\cvat`, http://localhost:8080 | La herramienta de anotación. Usa unos 2,9 GB de RAM, menos de lo estimado |
| Video de ejemplo | `data/samples/vehicles.mp4` | Para probar el sistema sin esperar los clips propios |
| Instrucciones | `README.md` | Instalación, uso, CVAT y acceso por Tailscale |

**Verificaciones del plan:**
- **YOLO:** el modelo vigente de Ultralytics es YOLO26, y elegiste YOLO26n.
- **TrackEval**, la librería planeada para las métricas de tracking, no funciona con numpy 2 sin ajustes. Se decide en la Fase 3a.

**Lo que no se pudo hacer:**
- **VisDrone:** Google Drive bloqueó la descarga automática por exceso de descargas. Hay que bajarlo desde el navegador; el link está en `AVANCE.md`.
- **Pasos con tus cuentas o los PCs de tus compañeros:** administrador de CVAT, Tailscale y repo en GitHub.

---

## 3. Fase 1 — Pipeline mínimo

### 3.1 Qué hace

```
video ──► YOLO26n + ByteTrack ──► detecciones con ID por cuadro ──► conteo por línea ──► events.csv + counts.csv
```

Se ejecuta con un comando:

```powershell
python -m vtia.cli count data\samples\vehicles.mp4 --lines configs\points\sample_vehicles.yaml
```

### 3.2 Arquitectura limpia (pragmática)

El código se separa en capas según qué tan "puro" es:

```
            ┌──────────────────────────────┐
            │  cli.py  (entrada)            │  arma todo y lo ejecuta
            └──────────────┬───────────────┘
                           │
   ┌───────────────────────▼──────────────────────┐
   │  adapters/  (lo que toca el mundo exterior)   │  YOLO, archivos YAML, CSV
   └───────────────────────┬──────────────────────┘
                           │ implementan
            ┌──────────────▼───────────────┐
            │  application/  (casos de uso) │  "procesar un video"
            └──────────────┬───────────────┘
                           │ usa
            ┌──────────────▼───────────────┐
            │  domain/  (reglas del negocio)│  clases, líneas, cómo se cuenta un cruce
            └──────────────────────────────┘
```

**La regla:** las flechas solo van hacia abajo. El dominio no sabe que existe YOLO, ni CSV, ni YAML. Eso da tres ventajas concretas para este proyecto:
1. **Cambiar de modelo no toca la lógica de conteo.** Cuando entren Faster R-CNN y RT-DETR (Fase 3), solo se agrega un adaptador nuevo.
2. **La lógica de conteo se prueba sin video ni modelo.** Las 6 pruebas corren en milisegundos con datos inventados.
3. **El tablero Streamlit (Fase 6) será otra entrada**, como `cli.py`, que reutiliza la misma aplicación sin duplicar lógica.

**Por qué es "pragmática":** solo hay una interfaz, el puerto `Tracker` en `application/process_video.py`, porque es lo único que tendrá varias implementaciones (YOLO, RT-DETR, Faster R-CNN). No se crearon interfaces, fábricas ni contenedores de inyección para cosas que tienen una sola versión.

| Archivo | Capa | Qué hace |
|---|---|---|
| `domain/classes.py` | Dominio | Las 6 clases y su equivalencia con las etiquetas de COCO |
| `domain/models.py` | Dominio | `Detection` (objeto en un cuadro), `Line` (línea virtual), `CrossingEvent` (un cruce) |
| `domain/counting.py` | Dominio | Decide cuándo un objeto cruzó una línea y en qué sentido |
| `application/process_video.py` | Aplicación | Caso de uso: recorre el video y acumula los cruces. Define el puerto `Tracker` |
| `adapters/ultralytics_tracker.py` | Adaptador | Ejecuta YOLO + ByteTrack y traduce su salida a `Detection` |
| `adapters/yaml_config.py` | Adaptador | Lee las líneas de `configs/points/*.yaml` y valida el formato |
| `adapters/csv_writer.py` | Adaptador | Escribe los CSV |
| `cli.py` | Entrada | Lee los argumentos, conecta las piezas e imprime el resumen |

### 3.3 Cómo se cuenta un cruce

1. **Punto de referencia:** de cada caja se usa el **centro del borde inferior**, donde el objeto toca el suelo. Con cámara elevada es más estable que el centro de la caja.
2. **Lado de la línea:** con un producto cruz se sabe si ese punto está a la izquierda o a la derecha de la línea.
3. **Cruce:** ocurre cuando el punto cambia de lado entre dos posiciones fuera de la línea **y** el movimiento pasa por el segmento dibujado, no por su prolongación. Así un vehículo del carril de al lado no cuenta en una línea que solo cubre un carril.
4. **Una vez por objeto y por línea:** si el punto tiembla justo sobre la línea, no se cuenta dos veces. El costo conocido: un objeto que de verdad cruza y se devuelve cuenta una sola vez. Está marcado en el código.
5. **Clase por mayoría:** el detector a veces cambia de opinión entre cuadros (en el video, el camión aparece como `truck` y como `bus`). La clase del cruce es la que más veces tuvo ese objeto, no la del cuadro exacto del cruce.

Las líneas se configuran por punto de cámara en YAML, con coordenadas en píxeles y un nombre para cada sentido (ver `configs/points/sample_vehicles.yaml`).

### 3.4 Resultados en el video de ejemplo

Autopista en 4K, 538 cuadros (21,5 s), procesada en la CPU de tu PC:

| Modelo | Cruces contados | Velocidad | Qué pasó |
|---|---|---|---|
| **YOLO26n** | 8 | 29,1 fps | Detectó dos veces el carro negro; luego el tracker le pasó ese ID al carro blanco de atrás, que ya no se contó |
| YOLO11n | 10 | 28,1 fps | Contó dos veces el carro negro: dos IDs cruzando en el mismo cuadro |

**Conteo real estimado: 9**, por revisión de los cuadros de cada evento; no revisé el video completo. Los 8 eventos de YOLO26n son correctos en clase y sentido. El camión del inicio no cuenta porque el video arranca con él ya pasando la línea.

**Lectura:** la lógica de conteo funciona. El error viene de **detecciones duplicadas** de un mismo vehículo, que confunden al tracker. No se corrigió a ciegas: se mide y ajusta en las Fases 3 y 5, con datos de referencia (umbral de confianza, BoT-SORT, supresión de duplicados).

### 3.5 Pruebas

`tests/test_counting.py`, 6 pruebas, todas pasan:
- cruza hacia abajo y hacia arriba con el sentido correcto;
- pasar al lado del segmento no cuenta;
- temblar sobre la línea cuenta una sola vez;
- detenerse justo sobre la línea y luego cruzar cuenta;
- la clase es la mayoritaria del track;
- dos objetos distintos cuentan por separado.

```powershell
python -m unittest discover -s tests
```

### 3.6 Lo que se dejó fuera a propósito

| Qué | Cuándo se agrega |
|---|---|
| Hora real de cada cruce (desde el nombre del archivo `P1_AAAAMMDD_HHMM.mp4`) | Fase 6, para los reportes por franja |
| Video de salida con cajas y línea dibujadas | Fase 6 (tablero) |
| Zonas (ROI), permanencia, alertas | Fase 5 |
| Faster R-CNN y RT-DETR | Fase 3 |
| Regla para unir al ciclista con la bici | Fase 3 (experimento E2) |

---

## 4. Qué sigue

1. **Equipo:** cerrar los pendientes de la Fase 0 (lista en `AVANCE.md`).
2. **Fase 3a:** construir el módulo de evaluación con VisDrone, cuando esté descargado.
3. **Fase 2:** protocolo de captura y guía de anotación, cuando definan ciudad, puntos y celulares.
