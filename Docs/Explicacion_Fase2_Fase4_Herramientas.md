# Qué hicimos: herramientas de las Fases 2 a 4 y el fondo del tablero

Fecha: 2026-10-07. La bitácora con el estado al día es [`AVANCE.md`](AVANCE.md).

---

## 1. Resumen

Quedó hecho **todo el código que no depende de sus datos**. Cuando lleguen los videos, el flujo completo ya existe y está probado:

```
grabar ──► anonymize ──► split ──► prelabel ──► CVAT (corregir) ──► evaluate / calibrate ──► train ──► evaluate
```

| Herramienta | Comando | Estado |
|---|---|---|
| Anonimización (cabezas, placas, parabrisas) | `anonymize` | Probada en los dos videos de ejemplo |
| Pre-anotación para CVAT | `prelabel` | Probada: genera el zip, se lee de vuelta y se evalúa |
| Lector de exportaciones de CVAT | `evaluate --cvat` | Probado con el zip anterior |
| Partición por sesión | `split` | Probada (ninguna sesión partida) |
| Regla del jinete (E2) | `--riders` | Implementada; se mide con sus datos |
| Calibración del umbral del tracker | `calibrate` | Corrida para los 4 modelos en VisDrone |
| Ajuste fino | `train` | **Ensayado en tu GPU: +62 % de mAP50** en secuencias no vistas |
| Fondo animado del tablero | — | Funcionando, verificado en el navegador |

Hay 41 pruebas automáticas y todas pasan. Todo sigue sin commitear.

---

## 2. Fondo animado del tablero

- **Qué es:** [upstream-fx](https://github.com/AmunM9/upstream-fx), con licencia MIT. Dibuja destellos de luz con WebGL que se abren alrededor del cursor.
- **Lo revisé antes de usarlo,** porque el repositorio se creó el mismo día y tiene 6 estrellas. El script no hace llamadas de red, no lee cookies ni almacenamiento y no usa `eval`; solo dibuja en un `<canvas>`.
- **Lo guardé dentro del proyecto** (`app/static/`, versión v0.1.0, con su licencia) en vez de cargarlo desde la CDN. Así funciona sin internet, nadie puede cambiarlo después y abrir el tablero no avisa a terceros.
- **El tablero pasó a tema oscuro:** el fondo es oscuro y el contenido va sobre un panel translúcido para que se lea.
- **Respeta el sistema:** se pausa en pestañas en segundo plano y respeta "reducir movimiento".

Para que funcionara corregí tres cosas, verificando cada una en Edge con capturas automáticas:
1. Streamlit 1.65 ya no inyecta CSS desde `st.markdown`; ahora se hace con `st.html`.
2. El efecto vive en un iframe que nacía con 0 px de alto; lo estiré a pantalla completa.
3. El desenfoque del panel "atrapaba" el fondo dentro del panel; se lo quité.

Además, el efecto reacciona al mouse: verifiqué que recibe los movimientos y abre el hueco alrededor del cursor.

---

## 3. Anonimización (`anonymize`)

**Qué difumina,** usando el mismo detector del proyecto (YOLO26s, confianza 0,15, baja a propósito):

| Objeto | Región | Por qué |
|---|---|---|
| Persona | El 28 % superior de su caja (la cabeza) | Funciona aunque la cara mida pocos píxeles, que es donde los detectores de rostros fallan |
| Carro, bus, camión, moto | La franja inferior de la caja (la placa) | Las placas siempre van abajo |
| Carro, bus, camión **cercanos** (≥ 250 px de alto) | El parabrisas | El conductor no se detecta como "persona" (ver más abajo) |

Además:
- Cada región se mantiene difuminada 3 cuadros más, para que una detección intermitente no deje ver una cara por un instante.
- No se guarda el audio, porque las voces también son datos personales.

**Por qué sin modelos de rostros o placas de terceros:**
- Los detectores de placas que se encuentran en internet son de usuarios anónimos.
- Están entrenados con placas de otros países.
- Varios tienen licencia AGPL.
- Los archivos `.pt` pueden ejecutar código al cargarse.

Si con sus clips reales se escapa algo, la alternativa es entrenar un detector propio de placas con CVAT.

**Lo que encontré revisando a ojo:** en la primera versión se veía **la cara del conductor del camión a través del parabrisas**. Por eso se agregó la franja del parabrisas, solo para vehículos grandes en la imagen; en los lejanos, los rostros de adentro no son identificables.

**Resultado en los ejemplos:** las cabezas de la plaza quedan difuminadas; solo quedan visibles figuras diminutas del borde, no identificables. En la autopista quedan cubiertas las placas y la cabina del camión.

**Tamaño de los archivos:** el códec de Windows generaba archivos 10 veces más pesados que el original (465 MB contra 35 MB). Cambié la escritura de video a **ffmpeg con x264**, mediante el paquete `imageio-ffmpeg` (BSD), que trae el ejecutable de ffmpeg incluido. Ahora los anonimizados pesan **menos que el original** (22 MB y 4,9 MB) y se ven en el navegador.

**Cada video genera también una hoja de control** (`*_check.jpg`, 12 cuadros). El protocolo de captura ahora exige revisarla antes de borrar el original. El programa **no borra nada solo**.

**Efecto en el ajuste fino:** los vehículos cercanos quedan en parte difuminados. Como todos los datos del proyecto pasan por la misma anonimización, entrenamiento y prueba quedan consistentes.

---

## 4. Pre-anotación y lectura de CVAT (`prelabel`, `evaluate --cvat`)

- **`prelabel`** genera, por cada clip, un zip en formato **MOT 1.1** que CVAT importa (*Upload annotations*):
  - usa **YOLO26x**, un modelo que **no** está en la comparación, para que la referencia no favorezca a ninguno de los evaluados (plan, Fase 2);
  - une el jinete con su vehículo, como pide la guía;
  - cada track lleva su clase mayoritaria;
  - escribe 1 de cada 5 cuadros y CVAT interpola el resto, así los anotadores corrigen menos cuadros clave.
- **El lector** toma la exportación de CVAT (zip o carpeta) y la convierte en datos de evaluación. La etiqueta `ignore` y las cajas marcadas como no consideradas pasan a ser regiones ignoradas.

**Cómo lo verifiqué:** pre-anoté la plaza, leí el zip como si fuera una exportación de CVAT y evalué YOLO26n y YOLO26s contra esa referencia. El flujo corre completo y da cifras con sentido; por ejemplo, YOLO26s saca mAP50 0,86 contra la referencia de YOLO26x. **No es una métrica real** (es un modelo contra otro), solo prueba el flujo.

**Falta un paso que necesita sus manos:** exportar el primer clip corregido desde su CVAT y correr `evaluate --cvat`, para confirmar que su versión de CVAT exporta todos los cuadros como espero.

---

## 5. Partición por sesión (`split`)

Toma los nombres de los clips (`P1_20261015_0705_s0312`) y arma una prueba fija más 5 grupos de validación cruzada:
- **La sesión completa va junta.** Los clips de una misma sesión se parecen demasiado; separarlos inflaría las métricas.
- **Estratifica por punto y franja** (mañana, mediodía, tarde), para que la prueba cubra todos los casos.
- **Es determinista:** con la misma semilla da siempre la misma partición.

Probado con 72 nombres de ejemplo (24 sesiones): ninguna sesión quedó partida, la prueba tiene una sesión por cada punto y franja, y los grupos quedan balanceados.

---

## 6. Regla del jinete (E2, `--riders`)

Los modelos preentrenados con COCO marcan al ciclista y al motociclista como "persona" en una caja aparte. La regla los une a su vehículo cuando la persona se superpone al menos un 40 % con la bici o moto y sus pies quedan dentro de su ancho; el parrillero también se une. Una persona parada junto a una bicicleta estacionada no se une.

**No se evaluó en VisDrone a propósito:** ahí los jinetes están anotados por separado, así que la regla "empeoraría" artificialmente. Se mide con sus datos, que siguen la convención de la guía (jinete + vehículo en una caja).

---

## 7. Calibración del umbral del tracker (`calibrate`, VisDrone)

| Modelo | Mejor umbral por HOTA | HOTA con ese umbral | HOTA con 0,6 (librería) |
|---|---|---|---|
| YOLO26n | 0,2 (empata con 0,3) | 0,286 | 0,196 |
| YOLO26s | **0,3** | 0,363 | 0,307 |
| RT-DETR-L | **0,4** | 0,381 | 0,344 |
| Faster R-CNN | **≥ 0,7** (borde de la grilla) | 0,352 | 0,346 |

- **Para YOLO26n se confirma el 0,3 que eligieron.** Iguala el mejor HOTA con 294 cambios de ID, contra 759 con 0,2.
- **YOLO26s también mejora con 0,3**, de HOTA 0,307 a 0,363.
- **Cada modelo pide un umbral distinto, por cómo reparte su confianza.** YOLO da confianzas bajas y prefiere umbrales bajos. Faster R-CNN da confianzas altas incluso a sus errores, así que prefiere umbrales altos, quizás por encima de 0,7. RT-DETR queda en el medio. Esto confirma que **el umbral se calibra por modelo** (plan §2.2) y que comparar todos con el mismo umbral sería injusto.
- No cambié ningún valor por defecto, porque decidieron calibrar con sus datos.

Las tablas completas están en `outputs/eval/VisDrone2019-MOT-val/calibration_*.md`.

---

## 8. Ajuste fino: ensayo en tu GPU (`train`)

**Montaje:** YOLO26n entrenado con 5 secuencias de VisDrone (477 imágenes, 1 de cada 5 cuadros) y validado con otras 2 que nunca vio. Fueron 50 épocas en **12,5 minutos**.

**Un obstáculo resuelto:** en Windows, ROCm 7.2.1 no logra compilar el kernel de MIOpen para la normalización por lotes en modo entrenamiento. La solución es apagar MIOpen solo al entrenar, lo que hace el código automáticamente. Es más lento, pero funciona. Está marcado para quitarse cuando AMD lo corrija.

**Resultado** sobre las 2 secuencias no vistas, con el mismo tracker (umbral 0,3) para ambos:

| Métrica | Sin ajustar | Ajustado |
|---|---|---|
| mAP50 | 0,163 | **0,264** (+62 %) |
| F1 macro | 0,159 | **0,293** (+84 %) |
| HOTA | 0,306 | **0,401** |
| IDF1 | 0,287 | **0,442** |
| MOTA | 0,171 | 0,168 |

| F1 por clase | Sin ajustar | Ajustado |
|---|---|---|
| Automóvil | 0,38 | **0,67** |
| Bicicleta | 0,08 | **0,53** |
| Peatón | 0,30 | **0,43** |
| Motocicleta | 0,00 | 0,13 |
| Bus | 0,00 | 0,00 |
| Camión | 0,19 | **0,00** |

**La lección más importante:** el bus y el camión **empeoran o quedan en cero** porque casi no aparecían en las secuencias de entrenamiento (los 264 buses estaban en las de validación). El modelo "olvida" lo que no ve. Para sus datos esto significa:
1. Asegurar buses y camiones en **todos** los grupos de la partición, con captura dirigida en corredores de carga.
2. Si aun así quedan pocos, usar el plan de contingencia de la propuesta: unir en "vehículo pesado".

Este modelo es solo el **ensayo** del proceso: se entrenó con video desde dron y **no** sirve para su cámara.

---

## 9. Otros cambios técnicos

- **Escritura de video con ffmpeg:** se usa para el video anotado y el anonimizado, y dio archivos ~20 veces más livianos.
- **El detector reconoce las clases de un modelo ajustado,** cuyas etiquetas ya son `pedestrian`, `car`, etc.
- **Nombres de los modelos ajustados:** quedan como `<corrida>-best`, para que dos entrenamientos no compartan la caché de detecciones por error.
- **Un solo decorador para los pasos de limpieza** (`PostProcessed`), que reemplaza al antiguo `Deduplicated`.

---

## 10. Qué falta (todo necesita sus datos o sus decisiones)

1. **Grabar, anonimizar y revisar las hojas de control.**
2. **Probar el primer zip real de CVAT** con `evaluate --cvat`.
3. **Anotar,** con la pre-anotación como punto de partida.
4. **Medir con sus datos:** E1 a E5, calibración y ajuste fino con validación cruzada.
5. **Decidir:**
   - el umbral por defecto de YOLO26s (la evidencia en VisDrone apunta a 0,3);
   - si la franja del parabrisas cubre lo necesario en sus puntos;
   - los casos dudosos de la guía de anotación.
