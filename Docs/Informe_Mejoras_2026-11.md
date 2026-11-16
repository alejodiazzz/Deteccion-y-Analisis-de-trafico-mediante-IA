# Informe: mejoras de anonimización, tablero en la nube, reutilización de detecciones y punto de cámara automático

Fechas: 2026-11-14 y 2026-11-15. Rama `feature/comparacion-modelos`, commits `57cc58c` a `6af3d79` (§10). La bitácora general es [`AVANCE.md`](AVANCE.md).

---

## 1. Resumen

- **Anonimización.** Las franjas fijas difuminaban casi todo el vehículo cercano, lo que sesgaba cualquier evaluación hecha sobre video anonimizado. Ahora, en los vehículos cercanos, un modelo de placas y uno de caras buscan la placa y el conductor reales, y solo eso se difumina. Lo que no encuentran se sigue cubriendo con franjas fijas: **la privacidad va primero**. El costo medido es **+15 % de tiempo en CPU**.
- **Tablero en la nube.** El tablero se publicó en **Streamlit Community Cloud**. Todo video subido se **anonimiza dentro del mismo proceso antes de analizarse**, y el original se borra enseguida. También hay un cuaderno de Colab para probarlo con GPU.
- **Reutilizar detecciones.** Las detecciones que se hacen al anonimizar se guardan y el análisis las reutiliza. Con el mismo modelo, el análisis de 8 s de video pasa de **17,3 s a 0,4 s**.
- **Hallazgo principal: el difuminado sesga la detección.** Sobre la copia anonimizada, YOLO26s detecta **82 % menos peatones, 65 % menos buses y 18 % menos carros** que sobre el original (§6.3). Toda evaluación de modelos debe hacerse sobre el video original o sobre detecciones del original, nunca sobre la copia anonimizada.
- **Punto de cámara automático.** Nuevo comando `suggest-point` y botón en el tablero: a partir de las trayectorias de los primeros minutos, propone líneas de conteo con su sentido permitido, zonas de calzada y zonas peatonales. Funciona bien con **cámara fija**. En `piloto_autopista` no sirve porque **la cámara se mueve**, lo que también afecta el conteo en ese clip.
- **Prueba piloto.** `evaluate --pilot` permite evaluar con videos externos anotados en CVAT, que no siguen el protocolo de particiones, y marca el informe como orientativo.

---

## 2. Modo piloto de evaluación (`evaluate --pilot`)

**Problema.** `evaluate` exige el protocolo completo: clips propios con nombre `P<punto>_<AAAAMMDD>_<HHMM>` y 5 particiones de validación. Mientras no hay grabaciones propias, no había forma de obtener una señal con videos externos.

**Qué se hizo.** La opción `--pilot`:
- evalúa los clips anotados en CVAT (`--cvat` + `--videos`) sin manifiesto de particiones;
- guarda los resultados en una carpeta con prefijo `pilot_` (`outputs/eval/pilot_<fuente>`);
- agrega al informe en Markdown y en PDF la advertencia de que los resultados son orientativos y no son la prueba fija;
- marca el JSON con `dataset_type: external_pilot_cvat_mot`.

No se puede combinar con `--splits` ni `--partition`. Así los resultados del piloto no se confunden con los de la prueba fija.

**Pruebas.** Hay dos pruebas nuevas: una verifica que se evalúen clips externos sin manifiesto, y la otra que el modo piloto no se pueda mezclar con el protocolo de particiones.

---

## 3. Anonimización

### 3.1 Punto de partida

La anonimización ubicaba cabezas, placas y parabrisas con **proporciones fijas** de las cajas que ya encuentra el detector (YOLO26s): la cabeza en la parte alta de la persona, la placa en una franja baja del vehículo y el parabrisas en una franja alta de los vehículos cercanos. Es barato y no depende de que se vea la placa o la cara, pero las franjas eran muy grandes:

| Parámetro | Original | Primer ajuste (`680eabb`) | Final (`e8647c3`) |
|---|---|---|---|
| Franja de placa del carro (alto × ancho de la caja) | 40 % × 80 % | 22 % × 55 % | **40 % × 60 %** (respaldo) |
| Franja de placa de bus y camión | 30 % × 70 % | 18 % × 50 % | **30 % × 55 %** (respaldo) |
| Franja de placa de moto | 45 % × 100 % | 30 % × 60 % | **45 % × 70 %** (respaldo) |
| Parabrisas (franja del alto; recorte lateral) | 8–55 %; 5 % | 15–45 %; 15 % | 15–45 %; 15 % (respaldo) |
| Margen alrededor de cada zona | 15 % | 8 % | 8 % (franjas) y 25 % (placas y caras encontradas) |
| Confianza del detector | 0,15 | 0,15 | **0,10** (§6) |

Con los valores originales, en un carro cercano la franja de placa y la del parabrisas, ya agrandadas por el margen, se superponían y **cubrían prácticamente todo el vehículo**.

### 3.2 Primer ajuste y por qué no bastó

Se estrecharon las franjas (columna "Primer ajuste"). Una prueba automática exige que un vehículo cercano quede difuminado en menos del 50 %.

Al revisar a ojo `piloto_autopista`, la franja estrecha **dejaba ver placas**. Con una cámara alta, la caja del vehículo incluye el techo, y la placa trasera queda a ~70 % de la altura de la caja, por encima de una franja que solo cubre el parachoques. Una franja fija no puede ser a la vez estrecha y segura para todos los ángulos de cámara.

### 3.3 Modelos de placas y caras

En los vehículos cercanos, dos modelos pequeños buscan la placa y el conductor reales:

| Uso | Modelo | Licencia | Tamaño |
|---|---|---|---|
| Placas | YOLOv11n de [morsetechlab](https://huggingface.co/morsetechlab/yolov11-license-plate-detection), afinado con ~10.000 imágenes de placas | AGPL-3.0 (la misma de Ultralytics, que el proyecto ya usa) | 5,5 MB |
| Caras de conductores | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) de OpenCV Zoo (versión 2026may, con tamaño de entrada dinámico, que pide OpenCV 5) | MIT | < 1 MB |

Ninguno se entrenó con placas colombianas ni con estas cámaras. Por eso se mantiene la revisión a ojo con las hojas de contacto (`*_check.jpg`).

**Reglas:**
- **Placas:** se buscan dentro de cada vehículo de al menos 60 px de alto. En los más pequeños la placa mide pocos píxeles y basta la franja.
- **Conductores:** se buscan en el 65 % superior de carros, buses y camiones de al menos 250 px de alto.
- **Peatones:** se mantiene la franja de la cabeza (28 % superior de la caja). Ya es pequeña y no depende de que se vea la cara.
- **Respaldo:** si el modelo no encuentra placa o conductor, se aplica la franja fija, que volvió a ser profunda (columna "Final"). Con cámara alta cubre la placa, y un vehículo cercano queda difuminado en menos del 60 % incluso en ese caso. **La privacidad va primero.**

**Medidas para abaratar:**
1. **Solo vehículos cercanos.** Los modelos trabajan sobre recortes de esos vehículos, no sobre el cuadro entero.
2. **Cada 4 cuadros.** Entre búsquedas, lo encontrado se expresa relativo a la caja del vehículo y la sigue. El vehículo se reconoce por superposición con su caja anterior (IoU ≥ 0,3). Un vehículo que aparece entre dos búsquedas recibe la franja fija hasta la siguiente.
3. **Placas a 320 px.** El modelo de placas trabaja con imágenes de 320 px; la prueba de tamaños está en §3.4.

**Opciones nuevas en la CLI:** `anonymize --no-refine` usa solo las franjas (más rápido, difumina más) y `--refresh N` cambia cada cuántos cuadros corren los modelos. Los modelos se descargan solos a `models/`, que no se sube a git.

### 3.4 Resultados

Se usaron 8 s de cada clip del piloto, en CPU (este PC no tiene GPU NVIDIA):

| Clip | Solo franjas | Con modelos | Placa encontrada (vehículos cercanos) | Conductor encontrado (cabinas) |
|---|---|---|---|---|
| `piloto_autopista` (cámara alta, día) | 34,8 s | 40,0 s (**+15 %**) | 542 / 1.021 (53 %) | 0 / 11 |
| `piloto_noche` (1080p, noche) | — | 149,4 s | 441 / 821 (54 %) | 5 / 82 |
| `piloto_cruce` (bicis y motos de lado) | — | 47,9 s | 1 / 86 | 0 / 2 |

**Tamaño de imagen del modelo de placas** (`piloto_autopista`, solo el tiempo del modelo de placas):

| imgsz | Placas encontradas | Tiempo |
|---|---|---|
| **320 (elegido)** | 53 % | 20,9 s |
| 480 | 60 % | 44,5 s |
| 640 | 52 % | 77,2 s |

Con 480 se gana poco a cambio del doble de tiempo, y con 640 no mejora. Muchos de los vehículos sin placa encontrada probablemente no la muestran: están de lado, lejos o tapados por otro.

**Revisión a ojo:**
- En los cuadros ampliados de `piloto_autopista` y `piloto_noche`, las placas quedaron cubiertas, ya sea por el modelo o por la franja de respaldo.
- Con los modelos, el vehículo queda casi intacto y solo se difumina la placa.
- **De noche** el modelo de caras casi nunca encuentra conductores (faros, oscuridad, reflejos), así que casi siempre se aplica la franja del parabrisas, que tapa buena parte del carro cercano.

---

## 4. Tablero en Streamlit Community Cloud y en Colab

### 4.1 Anonimizar todo video subido

**Qué se hizo.** Al subir un video al tablero:
1. El original se guarda solo en una carpeta temporal.
2. Se anonimiza (§3), con una barra de progreso.
3. La carpeta temporal se borra, con el original dentro.

El análisis, la comparación de modelos y las descargas usan **solo la copia anonimizada**. En la barra lateral se pueden revisar cuadros de muestra. Si el proceso se interrumpe, no queda guardado un video a medio difuminar. Los videos elegidos de `data/` no se tocan, porque se asume que ya pasaron por `anonymize`.

**Advertencia de privacidad.** Aunque el original se borre de inmediato, en la nube el video **sin difuminar viaja a los servidores de Streamlit o de Google, fuera de Colombia** (Ley 1581 de 2012). Para videos sensibles, lo correcto es anonimizarlos antes en el PC servidor con `python -m vtia.cli anonymize`.

### 4.2 Despliegue

| Archivo | Para qué |
|---|---|
| `app/requirements.txt` | Community Cloud lo lee antes que el de la raíz. Instala las mismas dependencias, pero con **torch solo para CPU**: la versión de Linux por defecto trae ~2,5 GB de librerías CUDA |
| `packages.txt` | Librerías del sistema que necesita OpenCV (`libgl1`, `libglib2.0-0`) |

La app usa la rama `feature/comparacion-modelos`, el archivo `app/dashboard.py` y **Python 3.12**. Un primer intento falló porque se desplegó sin estos archivos y con Python 3.14. La versión de Python no se puede cambiar después de desplegar: hay que borrar la app y crearla de nuevo.

### 4.3 Cuaderno de Colab

`notebooks/dashboard_colab.ipynb` sigue estos pasos:
1. Clona el repositorio en una T4.
2. Instala las dependencias sin reemplazar el pandas de Colab.
3. Prueba el conteo con un video de ejemplo.
4. Abre el tablero con un enlace temporal de [bore](https://github.com/ekzhang/bore).
5. Al final, apaga el tablero y el enlace.

El enlace es **público** mientras el tablero esté encendido: no debe compartirse y hay que apagarlo con la última celda.

---

## 5. Reutilización de detecciones

### 5.1 Qué se hizo

Cuando se sube un video al tablero, YOLO26s lo recorre completo para anonimizarlo, y luego el modelo elegido lo recorre otra vez para el análisis. Ahora:

1. `anonymize_file` guarda junto a la copia anonimizada las detecciones de YOLO26s hechas sobre el **original** (`<video>.detections.jsonl` y su `.meta.json`). El archivo solo se escribe si la pasada termina completa.
2. `analyze_file` las reutiliza solo si eso equivale exactamente a correr el modelo: el mismo modelo, el mismo tamaño de imagen, la misma cantidad de cuadros y una confianza pedida mayor o igual a la guardada. En ese caso filtra las detecciones por la confianza pedida. Si alguna condición no se cumple, detecta como siempre.
3. Para que coincidan, la confianza de la anonimización bajó de 0,15 a **0,10**, que es la del análisis. Eso difumina un poco más, nunca menos.
4. El informe PDF indica cuándo se usaron detecciones reutilizadas.

### 5.2 Resultado

En 8 s de `piloto_autopista`, el análisis con YOLO26s tarda **17,3 s detectando y 0,4 s reutilizando**. El ahorro solo aplica a YOLO26s, el modelo de la anonimización; el tablero lo indica junto al selector de modelo.

---

## 6. El difuminado sesga la detección

### 6.1 Medición

Se compararon las detecciones de YOLO26s sobre el **mismo clip** (8 s de `piloto_autopista`, confianza ≥ 0,25):

| Clase | Original | Anonimizado | Diferencia |
|---|---|---|---|
| Carro | 4.050 | 3.326 | −18 % |
| Bus | 1.311 | 458 | **−65 %** |
| Camión | 1.166 | 1.528 | +31 % |
| Peatón | 85 | 15 | **−82 %** |
| Moto | 33 | 12 | −64 % |
| Bicicleta | 30 | 2 | −93 % |

El aumento de camiones probablemente se debe a buses con el parabrisas difuminado que pasan a clasificarse como camiones; no se verificó. Personas, motos y bicicletas casi desaparecen porque la cabeza difuminada es una parte grande de una caja pequeña.

### 6.2 Consecuencias

- **Toda evaluación de modelos** (`evaluate`, la prueba fija y el piloto) debe hacerse sobre el video **original** o con detecciones reutilizadas del original, nunca sobre la copia anonimizada. Las anotaciones de CVAT sí pueden hacerse sobre la copia anonimizada, porque la geometría es la misma.
- El análisis de un video subido al tablero, con YOLO26s, ya usa detecciones del original (§5).
- Analizar una copia anonimizada con **otro modelo** sigue teniendo este sesgo. Conviene tenerlo presente al leer resultados en la nube con YOLO26n, RT-DETR o Faster R-CNN.

---

## 7. Sugerencia automática del punto de cámara

### 7.1 Problema

Cada punto de cámara necesita un YAML (`configs/points/`) con líneas de conteo, zonas y reglas en **píxeles**. Hasta ahora se armaba a mano: se sacaba un cuadro, se abría en un editor de imágenes, se leían las coordenadas y se escribían en el YAML. Es lento y propenso a errores.

### 7.2 Método (`vtia/domain/scene.py`)

1. **Trayectorias.** Se detectan y siguen los objetos durante los primeros 2 minutos (`--seconds`), uniendo los motociclistas y ciclistas con su vehículo. Si el video se anonimizó en el proyecto, se reutilizan sus detecciones.
2. **Vehículos en movimiento.** Se usan solo trayectorias que avanzan al menos 2 alturas de su caja y van bastante rectas: desplazamiento / recorrido ≥ 0,6, medido sobre ~10 puntos espaciados para que el temblor de las cajas en un trancón no cuente.
3. **Flujos.** Se agrupan primero por **dirección** (rumbos a menos de 35°) y luego por **calzada**: las trayectorias cuyos recorridos se tocan sobre una cuadrícula del ancho de un vehículo comparten calzada. Este criterio no se confunde con la perspectiva, donde los carriles convergen hacia el horizonte. Un flujo con menos del 10 % de las trayectorias del mayor se descarta como ruido (cambios de carril, cambios de ID del tracker).
4. **Línea de conteo.** Cada flujo recibe una línea perpendicular a su dirección, donde más trayectorias son visibles, con el ancho de la calzada. Se orienta de modo que su propio sentido sea `to_right` y el **permitido**; el sentido contrario genera la alerta de contraflujo. El nombre sale de la dirección: `toward_camera` / `away_from_camera` si el movimiento tiene una componente vertical clara, porque con perspectiva una vía que se aleja sube en diagonal; si no, `left_to_right` / `right_to_left`.
5. **Zonas de calzada.** Para cada flujo, la envolvente convexa de los puntos donde sus vehículos tocan el suelo.
6. **Zonas peatonales.** Personas que caminan (≤ 2 alturas de su caja por segundo), agrupadas sobre una cuadrícula. Solo cuentan las celdas que recorrieron **al menos 2 personas distintas**, para que trayectorias sueltas no se encadenen en una mancha. Una "persona" que está casi toda dentro de una calzada y no la cruza (quieta o avanzando con el tráfico) se toma como **jinete** de una moto o bicicleta. Una que la cruza se toma como peatón en un paso peatonal.
7. **Salida.** El YAML lleva un encabezado que explica que es una propuesta, junto con una **imagen `.png`** con las líneas, su sentido y las zonas dibujadas sobre el video. Las zonas no traen reglas (clases prohibidas, permanencia, ocupación): se dejan para quien revise.

**Uso:** `python -m vtia.cli suggest-point VIDEO --out configs/points/<punto>.yaml`. En el tablero, el desplegable «Sugerir punto de cámara» guarda `sugerido_<video>.yaml`, que aparece de inmediato en «Punto de cámara», y permite descargarlo.

### 7.3 Resultados

| Video | Resultado |
|---|---|
| `vehicles.mp4` (autopista de ejemplo, cámara fija, 4K) | **Correcto**: las dos calzadas con su forma, sus sentidos (izquierda hacia la cámara, derecha alejándose) y una línea en cada una |
| `piloto_cruce` (cámara fija) | Un flujo con la zona de la calle y su línea. No propone zonas peatonales porque en esos 2 minutos casi no hay personas caminando, solo ciclistas y motociclistas |
| `piloto_autopista` | **No sirve**: la cámara se mueve (el encuadre cambia hacia el cuadro 86), así que las trayectorias mezclan el movimiento de los vehículos con el de la cámara |

**Tiempo:** en este PC (CPU), unos 75 s para 30 s de video si hay que detectar, y unos 9 s si reutiliza las detecciones de la anonimización.

**Ajustes hechos con el clip real.** En `piloto_autopista`, la primera versión propuso flujos de ruido, partió una calzada en tres y creó una zona peatonal sobre la vía con motociclistas del trancón. Cada caso se convirtió en una prueba automática y se corrigió: la rectitud se mide sobre puntos espaciados, las calzadas se separan por conectividad, los flujos pequeños se descartan y se aplica la regla de jinetes. Aun así, ese clip no sirve por el movimiento de la cámara.

---

## 8. Hallazgos

1. **El difuminado sesga la detección** (§6). Las evaluaciones se hacen sobre el original o sobre detecciones del original.
2. **En `piloto_autopista` la cámara se mueve.** Las líneas fijas, el conteo y la sugerencia de punto no son fiables en ese clip. Hay que tenerlo en cuenta al leer los resultados de `evaluate --pilot`.
3. **Con cámara alta, la placa queda en la parte media de la caja.** Una franja fija estrecha no la cubre (§3.2).
4. **De noche casi no se detectan conductores**, así que casi siempre se aplica la franja del parabrisas y el carro cercano queda bastante difuminado (§3.4).
5. **Reutilizar las detecciones** ahorra una pasada completa y además elimina el sesgo del difuminado en el análisis (§5).

---

## 9. Limitaciones y pendientes

**Limitaciones:**
- Los modelos de placas y caras **no se validaron con placas ni cámaras colombianas**. Los porcentajes de §3.4 dicen cuántas veces encontraron algo, no cuántas acertaron, y la revisión a ojo cubrió pocos cuadros.
- Las mediciones son de **8 s por clip** en CPU. Los tiempos en GPU (Colab) o en Streamlit Cloud no se midieron.
- La sugerencia de punto se probó con **tres videos**, y solo dos con cámara fija. No detecta por sí sola si la cámara se mueve.
- En Streamlit Cloud, los puntos sugeridos se guardan en un disco que se borra al reiniciar la app: hay que descargar el YAML.

**Pendientes:**
1. **Prueba piloto completa:** preanotaciones en Colab, corrección en CVAT y `evaluate --pilot` con el video **original** (§6).
2. **Revisar a ojo los 4 clips del piloto ya anonimizados en Colab**, sobre todo `piloto_noche`.
3. **Franja del parabrisas de noche.** Decidir si se omite, porque con faros y oscuridad los conductores no son identificables. Hoy se aplica por precaución.
4. **Aviso de cámara en movimiento** en la sugerencia de punto: comparar el fondo entre cuadros y advertir.
5. **Otras técnicas para aligerar los modelos**, que aún no se aplican: detectar 1 de cada 2–3 cuadros, FP16 en GPU, exportar a ONNX/OpenVINO para CPU y recortar a las zonas de interés. Cada una debe medirse con la evaluación antes de adoptarla.

---

## 10. Commits y cómo reproducir

| Commit | Cambio |
|---|---|
| `57cc58c` | Modo `--pilot` de `evaluate` |
| `680eabb` | Primer ajuste de las franjas de anonimización |
| `71ea73a` | Tablero para Streamlit Community Cloud y anonimización de videos subidos |
| `6184db1` | Cuaderno de Colab para el tablero |
| `e8647c3` | Modelos de placas y caras; franja de respaldo profunda |
| `fa101c3` | Pruebas de anonimización en los clips de noche y cruce |
| `58ffeb3` | Reutilización de detecciones; confianza de anonimización 0,10 |
| `6af3d79` | Sugerencia automática del punto de cámara |

```bash
# Anonimizar (con modelos de placas y caras; --no-refine = solo franjas)
python -m vtia.cli anonymize data/raw --out data/anon

# Sugerir un punto de cámara (revisar la imagen .png que queda al lado)
python -m vtia.cli suggest-point data/anon/<clip>.mp4 --out configs/points/<punto>.yaml

# Evaluación piloto: siempre sobre el video original (§6)
python -m vtia.cli evaluate --cvat data/annotations --videos data/raw --pilot --models yolo26n.pt yolo26s.pt
```

Pruebas: 153 en total (`python -m unittest discover -s tests`). Las nuevas son `tests/test_scene.py` y `tests/test_detection_log.py`, además de las ampliadas en `tests/test_data_prep.py`.
