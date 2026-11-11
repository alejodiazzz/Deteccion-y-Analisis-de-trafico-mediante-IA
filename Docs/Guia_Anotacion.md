# Guía de anotación — BORRADOR

Versión 0.1 · 2026-10-07 · Fase 2 del plan.
Lo marcado **[ABIERTO]** lo decide el equipo; lo marcado **[PROPUESTA]** es una recomendación que hay que confirmar. Cuando no quede nada marcado, la guía pasa a versión 1.0 y no se cambia a mitad de la anotación. Si hay que cambiarla, se re-anota lo ya hecho.

---

## 1. Para qué sirve

Que los tres anotadores dibujen **exactamente lo mismo** ante la misma escena. Las métricas del proyecto solo son tan buenas como la consistencia de las anotaciones: un desacuerdo entre anotadores se ve, en las métricas, como un error del modelo.

## 2. Configuración en CVAT

- **Proyecto:** las 6 clases + `ignore`. El JSON de etiquetas está en el README.
- **Una tarea por clip.** Siempre se sube el video **anonimizado**, nunca el crudo.
- **Modo track** (no "shape"): cada objeto es un track con un ID que se mantiene mientras está a la vista.
- **Interpolación:** se dibujan cuadros clave y CVAT interpola entre ellos.
- **Pre-anotación (implementada):** `python -m vtia.cli prelabel data\anon --out data\prelabels` genera un zip por clip con tracks propuestos por YOLO26x, con el jinete ya unido a su vehículo. En la tarea de CVAT: *Actions > Upload annotations > MOT 1.1*. El anotador corrige, borra lo que sobra y agrega lo que falta; no dibuja desde cero.

## 3. Clases

| Etiqueta | Qué incluye | Qué NO incluye |
|---|---|---|
| `pedestrian` | Persona caminando, corriendo, detenida o sentada en vía, andén o cruce | Ciclistas y motociclistas (van con su vehículo) |
| `bicycle` | Ciclista **y** bicicleta en **una sola caja** | — |
| `motorcycle` | Motociclista + parrillero + moto en **una sola caja**; moto estacionada sin conductor | Motocarros **[ABIERTO]** |
| `car` | Carro, taxi, camioneta (SUV, pickup), campero | Furgonetas de pasajeros **[ABIERTO]** |
| `bus` | Bus, buseta, microbús, bus articulado | Chiva **[ABIERTO]** |
| `truck` | Camión, volqueta, furgón, tractomula, camión recolector | Pickups (son `car`) |
| `ignore` | Región que no se evalúa (ver §5) | — |

**Casos frontera sin decidir [ABIERTO]:**
- Motocarro o mototaxi: ¿`motorcycle`, `car` o `ignore`?
- Chiva: ¿`bus` o `ignore`?
- Furgoneta de pasajeros (tipo Hiace): ¿`bus` (si es colectivo) o `car`?
- Carreta o zorra (tracción animal o humana), patineta o scooter: ¿`ignore` o no se anotan?
- Persona empujando una bicicleta: ¿`pedestrian` o `bicycle`?
- Bicicleta estacionada sin ciclista: ¿se anota?
- Vehículos de emergencia y policía: **[PROPUESTA]** por su forma (patrulla = `car`, ambulancia tipo furgón = `truck`).

## 4. Cómo se dibuja la caja

- **Ajustada** al objeto: sin espacio vacío alrededor y sin cortar partes.
- **Objetos ocluidos [PROPUESTA]:** se encierra solo la parte **visible** y se marca la casilla `occluded` de CVAT. Así se anotan también los datasets con los que se entrenaron los modelos.
- **Unidad conductor + vehículo:** la caja cubre a ambos (de la cabeza del conductor a la rueda más baja).
- **Tamaño mínimo [PROPUESTA]:** objetos de menos de 15 px de alto se cubren con una región `ignore` en vez de anotarse uno por uno.

## 5. Regiones `ignore`

Se dibuja un rectángulo `ignore` sobre:
- multitudes o filas de vehículos tan apretadas que no se pueden separar;
- objetos demasiado pequeños o lejanos (§4);
- objetos fuera del esquema de clases, si se decide así en §3.

Lo que cae mayoritariamente dentro de una región `ignore` no cuenta ni como acierto ni como error, ni para el modelo ni para la anotación. El código ya lo aplica así: lo validamos con VisDrone.

## 6. Reglas del track

- **Un objeto = un track** desde que entra hasta que sale de la imagen.
- **La clase no cambia** dentro de un track. Si te equivocaste, se corrige el track entero.
- **Oclusión breve** (pasa detrás de un bus): el track continúa con el mismo ID; en los cuadros donde no se ve, se marca `outside`.
- **Sale y vuelve a entrar** **[PROPUESTA]:** si es claramente el mismo objeto, mismo ID; si hay duda, track nuevo.
- **Cuadros clave:** el primero y el último del objeto, más uno cada 5–10 cuadros y donde cambie de velocidad o dirección. Antes de cerrar el track, revisar que la interpolación no se despegue del objeto.

## 7. Calidad

- **Doble anotación:** el 10 % de los clips lo anotan dos personas por separado para medir el acuerdo (IoU y clase).
- **Revisión cruzada:** cada clip lo revisa alguien distinto de quien lo anotó, antes de darlo por terminado.
- **Tiempo por clip:** cada uno registra en la planilla cuánto tardó. Con los primeros 5 clips se re-planifica.
- **Formato de exportación:** "MOT 1.1" de CVAT (*Actions > Export task dataset*), guardado como `data\annotations\<clip>.zip`. El lector está implementado y probado (`vtia/adapters/cvat_mot.py`); igual conviene exportar el primer clip real y correr `evaluate --cvat` antes de anotar en masa.

## 8. Checklist del anotador

- [ ] El video es la versión anonimizada
- [ ] Todo objeto de las 6 clases tiene track, o está bajo una región `ignore`
- [ ] Ciclistas y motociclistas van en una sola caja con su vehículo
- [ ] Ningún track cambia de clase
- [ ] La interpolación sigue al objeto en todo el track
- [ ] Tiempo registrado en la planilla
