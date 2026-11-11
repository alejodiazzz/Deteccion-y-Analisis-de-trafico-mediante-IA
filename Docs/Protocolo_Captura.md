# Protocolo de captura — BORRADOR

Versión 0.1 · 2026-10-07 · Fase 2 del plan.
**[ABIERTO]** lo decide el equipo; **[PROPUESTA]** es una recomendación que hay que confirmar.

---

## 1. Qué se graba

| Tipo | Cantidad | Duración | Uso |
|---|---|---|---|
| Sesiones largas | 2–4 por punto | 15–30 min | Reportes (franja, permanencia, alertas) y conteo manual de referencia |
| Clips cortos | 50–100 en total | 10–20 s | Anotación con trayectorias: evaluación y ajuste fino |

**[PROPUESTA] Los clips cortos se recortan de las sesiones largas** en vez de grabarse aparte: una sola salida a campo sirve para ambos fines.
Consecuencia importante: dos clips de la misma sesión se parecen mucho (misma luz, mismos vehículos). Por eso **la partición de datos se agrupa por sesión, no por clip**: todos los clips de una sesión van juntos a entrenamiento o juntos a prueba. Si no, las métricas salen infladas.

## 2. Puntos de captura **[ABIERTO]**

- Ciudad: **[ABIERTO]**
- Puntos (2–3): **[ABIERTO]**

Criterios para elegirlos:
- **Variedad de clases:** al menos uno con buses y camiones (corredor de carga o ruta de transporte público); hacen falta ≥ 15 de cada uno en la prueba.
- **Altura de 3 a 6 m,** mirando la vía en diagonal hacia abajo: puente peatonal, segundo piso o terraza con permiso.
- **Vía visible de lado a lado,** para poder trazar la línea virtual sin que los vehículos se tapen por completo.
- **Sin contraluz** en las franjas de grabación (revisar hacia dónde da el sol).
- **Seguridad** del equipo y de quien graba; sin obstruir el paso.
- **Espacio público.** Si se graba desde una propiedad privada, con permiso escrito del dueño.

Por cada punto se guarda una **foto de referencia del encuadre**, para repetirlo igual en cada sesión. La línea virtual y las zonas del YAML del punto dependen de ese encuadre.

## 3. Equipo y configuración del celular

- Celular **[ABIERTO: modelos]**, trípode o soporte fijo, batería externa y espacio libre (unos 6 GB por hora a 1080p).
- **Resolución y fps [PROPUESTA]:** 1080p a 30 fps, en el mismo ajuste para todo el proyecto.
- **[PROPUESTA]:**
  - Bloquear el enfoque y la exposición (mantener el dedo sobre la pantalla en la mayoría de celulares), para que la imagen no "respire" cuando pasa un bus.
  - Desactivar la estabilización electrónica si recorta o deforma la imagen; con trípode no hace falta.
  - Celular en modo avión: sin llamadas ni notificaciones que corten la grabación.
- **Horizontal,** sin zoom digital y sin moverlo durante la grabación.

## 4. Franjas horarias

**[PROPUESTA]** Tres franjas diurnas por punto, cubriendo horas pico:

| Franja | Horario |
|---|---|
| Mañana | 6:30–9:00 |
| Mediodía | 11:30–14:00 |
| Tarde | 16:30–18:30, antes de que oscurezca |

Sin visión nocturna (fuera de alcance). Se puede grabar con lluvia leve o media; se anota en la planilla.

## 5. Nombre de los archivos

```
P<punto>_<AAAAMMDD>_<HHMM>.mp4      ej.: P1_20261015_0705.mp4
```

`HHMM` es la hora local a la que **empieza** la grabación. El sistema saca la franja horaria del nombre, así que no puede tener errores. Los clips cortos agregan el segundo de inicio dentro de la sesión: `P1_20261015_0705_s0312.mp4`.

## 6. Planilla de sesiones

Una fila por sesión (hoja compartida o CSV):

| archivo | punto | fecha | hora_inicio | duracion_min | clima | iluminacion | celular | altura_m | responsable | notas |
|---|---|---|---|---|---|---|---|---|---|---|

## 7. Procedimiento en sitio

1. Llegar, montar el trípode y comparar el encuadre con la foto de referencia del punto.
2. Bloquear enfoque y exposición; revisar que la vía se vea de lado a lado.
3. Grabar 20 s de prueba y verlos completos (nitidez, encuadre, sin cortes).
4. Grabar la sesión sin tocar el celular.
5. Llenar la fila de la planilla antes de irse.

## 8. Después de grabar: privacidad (Ley 1581 de 2012)

1. Copiar los videos por **cable** al PC servidor, a `data/raw/`. No pasan por WhatsApp, Drive ni correo.
2. **Borrar** los videos del celular cuando la copia esté verificada.
3. **Anonimizar** en el PC servidor: `python -m vtia.cli anonymize data\raw --out data\anon`. Difumina la cabeza de cada persona, la placa de cada vehículo y el parabrisas de los vehículos cercanos, y descarta el audio. Solo la versión anonimizada va a CVAT, Drive o Colab.
4. Verificar la anonimización mirando cada `*_check.jpg` (12 cuadros del video) y reproduciendo el video. Si algo se escapa, avisar antes de seguir. Recién entonces **borrar el crudo**.
5. Generar la partición por sesión: `python -m vtia.cli split data\anon --out data\splits.csv`.
6. **[ABIERTO]** Señalización en el punto ("zona de grabación con fines académicos") y si se necesita carta de la universidad o permiso de la alcaldía.

## 9. Puntos abiertos (resumen)

- [ ] Ciudad y 2–3 puntos
- [ ] Permisos y señalización
- [ ] Celulares a usar
- [ ] Confirmar 1080p30, franjas y clips recortados de sesiones
- [ ] Quién va a cada sesión (en parejas)
