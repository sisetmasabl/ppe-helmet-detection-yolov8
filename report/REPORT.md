# Detección de Cumplimiento de Normas de Seguridad (EPP/Casco) en Obras de Construcción con YOLOv8

**Énfasis: detección a corta y larga distancia (objetos pequeños), robustez ante oclusión y un verificador de cumplimiento basado en reglas.**

*Autor: [COMPLETAR NOMBRE]* · *Curso: [COMPLETAR CURSO]* · *Fecha: mayo 2026*

> **Nota:** El informe en **formato NeurIPS (PDF)** —el entregable principal— está en
> `report/neurips/main.pdf` (fuente LaTeX en `report/neurips/main.tex`). Este
> documento Markdown es la versión narrativa de trabajo y genera `report/REPORT.pdf`.

---

## Resumen (Abstract)

Presentamos un sistema de visión por computador para monitorear el cumplimiento del uso de Equipos de Protección Personal (EPP), específicamente el casco de seguridad, en obras de construcción. Entrenamos un detector **YOLOv8s** sobre el **Safety-Helmet-Wearing-Dataset (SHWD)** (uno de los datasets recomendados por la consigna), que incluye ejemplos negativos **explícitos** de "sin casco" (clase *head*) frente a "con casco" (clase *helmet*), con 7 581 imágenes y resolución de entrada de 640 px, partiendo de un backbone preentrenado en COCO. Sobre el conjunto de prueba obtenemos **mAP@0.5 = 0.942** y **mAP@0.5:0.95 = 0.614**. Más allá de la métrica agregada, analizamos la robustez estratificando el conjunto de prueba por **tamaño de objeto** (proxy de distancia) y por **nivel de oclusión** estimado de forma semiautomática. Implementamos además un **verificador de cumplimiento basado en reglas** que convierte las detecciones por fotograma en una tasa de cumplimiento y marca las violaciones, y discutimos las implicaciones éticas de desplegar un sistema de este tipo.

---

## 1. Motivación

La industria de la construcción concentra una proporción desmesurada de accidentes laborales mortales; los traumatismos craneoencefálicos por caída de objetos o caídas de altura están entre las causas principales, y el uso correcto del casco es una de las medidas de control más efectivas y baratas. La supervisión manual del cumplimiento del EPP es costosa, intermitente y propensa a error humano, especialmente en obras grandes con muchos frentes de trabajo simultáneos. Un sistema automático que procese flujos de cámaras de obra y señale en tiempo (casi) real a los trabajadores sin casco puede aumentar la cobertura de la supervisión y generar registros auditables.

El problema no es un caso estándar de detección de objetos. En condiciones reales aparecen tres dificultades que estructuran este trabajo:

1. **Detección de objetos pequeños / larga distancia.** Las cámaras de obra cubren áreas amplias; los trabajadores lejanos aparecen con muy pocos píxeles y la cabeza/casco es aún más pequeña. Es el foco principal del trabajo.
2. **Oclusión severa.** Los trabajadores quedan parcialmente ocultos tras maquinaria, andamios, materiales u otros trabajadores.
3. **Desbalance de clases.** En SHWD la clase mayoritaria es, de hecho, la cabeza descubierta (*head*); la clase con casco es minoritaria, lo que sesga las métricas agregadas y dificulta el aprendizaje de la clase crítica.

Nuestro objetivo es entrenar y evaluar críticamente un detector YOLOv8 bajo estas condiciones, con especial atención a la detección a larga distancia, y construir sobre él un módulo de verificación de cumplimiento utilizable y éticamente informado.

## 2. Trabajos Relacionados

La detección automática de equipos de protección personal (EPP) en obras de construcción ha avanzado rápidamente con el aprendizaje profundo. El trabajo temprano de Fang et al. demostró que un modelo Faster R-CNN podía señalar a trabajadores sin casco a partir de video de vigilancia de campo lejano, destacando la dificultad de detectar trabajadores distantes y de baja resolución [fang2018nonhardhat]. Los sistemas posteriores adoptaron detectores de una sola etapa para rendimiento en tiempo real: la familia YOLO, introducida por Redmon et al. [redmon2016yolo] y refinada mediante avances arquitectónicos como backbones CSP y cuellos PANet [bochkovskiy2020yolov4] e información de gradiente programable [wang2024yolov9], domina hoy la detección de cascos de seguridad. Hayat y Morgado-Dias, por ejemplo, construyeron un monitor de cascos basado en YOLO en tiempo real [hayat2022helmet]. Nuestro trabajo usa Ultralytics YOLOv8 [jocher2023yolov8], que no tiene publicación revisada por pares y por tanto se cita como software, y se evalúa sobre el Safety Helmet Wearing Dataset (SHWD) [njvisionpower2019shwd].

Tres retos recurren en este dominio. La detección de objetos pequeños de trabajadores distantes se aborda típicamente mediante fusión de características multiescala como las Feature Pyramid Networks [lin2017fpn], y se revisa exhaustivamente en literatura reciente que también cataloga remedios para oclusión y desbalance de clases [kang2025sodsurvey]. La oclusión severa detrás de maquinaria y andamios refleja el problema de oclusión en multitudes de la detección de peatones, donde la *repulsion loss* de Wang et al. logra una localización más robusta a oclusión [wang2018repulsion]. Finalmente, la rareza relativa de una de las clases plantea un problema de desbalance señalado a lo largo de estos trabajos. Nuestro proyecto se apoya en esta literatura, adaptando YOLOv8 para abordar conjuntamente los retos de objetos pequeños, oclusión y desbalance en el monitoreo de cumplimiento de cascos.

## 3. Metodología

### 3.1 Conjunto de datos
Usamos el **Safety-Helmet-Wearing-Dataset (SHWD)** [njvisionpower2019shwd], recomendado por la consigna por aportar ejemplos negativos explícitos de "sin casco". Convertimos sus anotaciones de Pascal VOC a formato YOLO (`src/prepare_shwd.py`) y respetamos los splits predefinidos (sin fuga de datos): **5 457 / 607 / 1 517 imágenes** para train/val/test. Mapeamos las dos clases a: *helmet* (cabeza **con** casco = CONFORME) y *head* (cabeza **sin** casco = VIOLACIÓN). La Tabla 1 resume la distribución de instancias; evidencia un **fuerte desbalance** hacia *head* (cabezas descubiertas, en buena parte de escenas con multitudes).

**Tabla 1. Distribución de instancias por clase y split (SHWD).**

| Clase | train | val | test |
|---|---|---|---|
| helmet (con casco) | 6 419 | 747 | 1 878 |
| head (sin casco) | 79 778 | 9 178 | 22 558 |
| imágenes | 5 457 | 607 | 1 517 |

La Figura 1 muestra esta distribución y la Figura 2 la de tamaños de bounding box: el **88 %** de las cabezas *head* caen en la categoría "pequeño" (lejano), lo que motiva fuertemente el análisis por distancia.

![Figura 1. Distribución de instancias por clase y split: la clase "head" (sin casco) domina (desbalance).](outputs/dataset_class_distribution.png)

![Figura 2. Distribución de tamaños de bounding box: la gran mayoría de objetos son "small" (lejanos), reto central de objetos pequeños.](outputs/dataset_bbox_size_distribution.png)

### 3.2 Modelo y backbone
Empleamos **YOLOv8s** (variante *small*, ~11 M de parámetros) inicializado con pesos preentrenados en COCO, conforme a la consigna que permite usar backbones preentrenados. YOLOv8 es **anchor-free**: predice directamente el centro y las dimensiones de cada caja sin un banco de anclas predefinido, lo que reduce el sesgo de escala y favorece la detección de objetos pequeños. Su *neck* tipo PAN-FPN fusiona características a múltiples escalas (P3–P5), mecanismo estándar para objetos pequeños [lin2017fpn].

### 3.3 Decisiones de entrenamiento (documentadas)
- **Resolución de entrada:** 640×640 px, equilibrio entre detalle para objetos pequeños y memoria de una GPU de 8 GB.
- **Anclas:** no aplica — arquitectura anchor-free.
- **Aumento de datos:** mosaic (=1.0) **activado**, con `close_mosaic=10` para **desactivarlo en las últimas 10 épocas** y estabilizar la localización final; volteo horizontal (p=0.5); variación HSV (h=0.015, s=0.7, v=0.4) para robustez a iluminación; `translate=0.1` y `scale=0.5` —este último simula variaciones de **distancia**, clave para el objetivo de larga distancia. Se desactivaron mixup y copy-paste.
- **Optimización:** optimizador y *learning rate* automáticos de Ultralytics (AdamW); 100 épocas; batch=8 y `workers=2` (limitados por la VRAM de 8 GB); semilla=42 para reproducibilidad; precisión mixta (AMP).

### 3.4 Verificador de cumplimiento basado en reglas
Como SHWD distingue directamente *helmet* (con casco) de *head* (sin casco), el verificador (`src/compliance_verifier.py`, implementado por el estudiante) opera por **clasificación directa**: cada detección por encima del umbral de confianza (0.25) se categoriza como CONFORME (*helmet*) o VIOLACIÓN (*head*); la **tasa de cumplimiento por fotograma** es `#helmet / (#helmet + #head)`, y se dibujan en verde las cajas conformes ("OK casco") y en rojo (trazo grueso) las violaciones ("SIN CASCO"), con un encabezado que muestra la tasa. El módulo también soporta un modo de **asociación espacial** casco↔persona para datasets sin clase "sin casco". Esta regla aborda directamente el reto de larga distancia: una cabeza lejana mal clasificada altera la tasa, que es justamente el caso relevante para el análisis de **falsos negativos** (§6).

### 3.5 Análisis de robustez (oclusión y distancia)
Como el dataset no incluye etiquetas de oclusión, particionamos el conjunto de prueba de forma **semiautomática**:
- **Distancia (tamaño):** clasificamos cada objeto por su tamaño lineal relativo `sqrt(w·h)` (normalizado): *small* (<0.08, lejano), *medium* (0.08–0.25), *large* (>0.25, cercano).
- **Oclusión:** para cada objeto calculamos el máximo IoU con cualquier otro objeto del mismo fotograma como proxy de oclusión mutua [wang2018repulsion]: *bajo* (<0.10), *medio* (0.10–0.35), *alto* (>0.35).

Para cada grupo reportamos el **recall@IoU0.5** en el punto de operación conf=0.25, además del mAP global y por clase.

## 4. Experimentos

**Hardware/Software:** GPU NVIDIA RTX 5060 Laptop (8 GB, arquitectura Blackwell sm_120), PyTorch 2.11.0 + CUDA 12.8, Ultralytics 8.4.56, Python 3.13, Windows 11. **Reproducibilidad:** semilla 42; el pipeline completo se ejecuta con un único comando, `python run_pipeline.py` (descarga SHWD vía gdown **sin credenciales**, convierte, entrena, evalúa y genera la demo). Tiempo de entrenamiento: **≈7 h 41 min** (27 695 s, ~4.6 min/época) para 100 épocas.

## 5. Resultados

### 5.1 Métricas globales y por clase
Métricas sobre el conjunto de **prueba** (1 517 imágenes, no usadas en entrenamiento):

| Clase | AP@0.5 | AP@0.5:0.95 | Precisión | Recall |
|---|---|---|---|---|
| helmet (con casco) | 0.932 | 0.726 | 0.914 | 0.904 |
| head (sin casco) | 0.952 | 0.502 | 0.936 | 0.911 |
| Global (media) | 0.942 | 0.614 | 0.925 | 0.908 |

mAP@0.5 = **0.942**, mAP@0.5:0.95 = **0.614**. Un resultado notablemente sólido. Obsérvese un contraste revelador: la clase *head* tiene **mejor AP@0.5 (0.952)** pero **peor AP@0.5:0.95 (0.502)** que *helmet* (0.726). La explicación es geométrica: las cabezas son objetos diminutos (88 % "small"), y en objetos pequeños la IoU es muy sensible a errores de pocos píxeles, por lo que la localización fina (umbrales de IoU altos) se degrada aunque la detección a IoU0.5 sea excelente. La Figura 3 (matriz de confusión) muestra que los errores se concentran en falsos fondos (objetos pequeños no detectados) más que en confusión entre clases.

![Figura 3. Matriz de confusión normalizada sobre el conjunto de prueba.](outputs/val_metrics/confusion_matrix_normalized.png)

### 5.2 Robustez por distancia (objetos pequeños = larga distancia)
Recall@IoU0.5 (conf=0.25) por tamaño de objeto (proxy de distancia):

| Grupo (tamaño) | Recall |
|---|---|
| small (lejano) | 0.927 |
| medium | 0.973 |
| large (cercano) | 0.978 |

La Figura 4 muestra la caída de recall del grupo cercano (*large*) al lejano (*small*): el recall **global** apenas baja de 0.978 a 0.927 —el modelo es notablemente robusto a la distancia gracias a la enorme cantidad de cabezas pequeñas vistas en entrenamiento—, pero el efecto es mucho más marcado en la **clase crítica *helmet***, que cae de **0.99 (cercano) a 0.81 (lejano)**. Es decir, los **cascos lejanos se detectan peor**, lo que en la lógica de cumplimiento puede convertir a un trabajador conforme en una falsa violación. Esto confirma que la detección a **larga distancia** sigue siendo el régimen más difícil para la clase que más importa.

![Figura 4. Recall por distancia (tamaño de objeto como proxy).](outputs/robustness_by_distance.png)

### 5.3 Robustez por oclusión
Recall@IoU0.5 por nivel de oclusión estimado (IoU mutuo):

| Grupo (oclusión) | Recall |
|---|---|
| bajo | 0.947 |
| medio | 0.879 |
| alto | 0.608 |

La Figura 5 muestra una **degradación monótona y nítida** del recall al aumentar la oclusión estimada (0.95 → 0.88 → **0.61**): en las escenas de multitud de SHWD el solape mutuo entre cabezas sí captura oclusión real, y el grupo de oclusión alta (poco frecuente, n=74) es con diferencia el más difícil. Nótese que aquí el proxy funciona mejor que en un dataset multiescala porque las cabezas son **uniformemente pequeñas**, de modo que el solape no está confundido con el tamaño; aun así, no captura la oclusión por estructuras estáticas (andamios, maquinaria), limitación que discutimos en §7.

![Figura 5. Recall por nivel de oclusión estimado.](outputs/robustness_by_occlusion.png)

### 5.4 Verificador de cumplimiento (demostración)
Aplicamos el verificador a **80 imágenes** del conjunto de prueba (no vistas en entrenamiento) y generamos un clip de video corto (`outputs/demo/demo_clip.mp4`) y un CSV de cumplimiento por fotograma (`outputs/demo/compliance_per_frame.csv`). **23 de 80** fotogramas presentan al menos una violación y la tasa de cumplimiento **media es del 88.4 %**, coherente con un subconjunto dominado por escenas de obra con trabajadores mayormente equipados. La Figura 6 muestra ejemplos anotados: el verificador marca correctamente en verde a los trabajadores con casco ("OK casco") y en rojo a los que no lo llevan ("SIN CASCO"), con la tasa de cumplimiento por fotograma en el encabezado.

![Figura 6. Demostración del verificador sobre imágenes de test no vistas (escena de obra con conformes en verde y violaciones en rojo).](outputs/demo/example_mixed.jpg)

![Figura 6 (cont.). Escena con múltiples trabajadores: cascos detectados (verde) y cabezas sin casco (rojo).](outputs/demo/example_violations.jpg)

## 6. Implicaciones Éticas

El despliegue de un sistema automático de vigilancia del cumplimiento de EPP conlleva tensiones éticas inseparables de las decisiones técnicas.

**Falsos negativos vs. falsos positivos.** El coste de los dos errores es marcadamente asimétrico. Un **falso negativo** (un trabajador sin casco que el sistema no detecta) significa una violación de seguridad no señalada: si la organización confía en el sistema y reduce la supervisión humana, puede traducirse directamente en un accidente grave o mortal. Un **falso positivo** (marcar como infractor a un trabajador que sí cumple) erosiona la confianza, puede derivar en sanciones injustas y, si se repite, provoca "fatiga de alerta" que lleva a ignorar todas las alertas —reintroduciendo el riesgo de los falsos negativos. Nuestro análisis por distancia y oclusión muestra que los falsos negativos se concentran en los trabajadores **lejanos** y **ocluidos**, que son también los más expuestos; por ello recomendamos operar con un **umbral de confianza bajo** (priorizando recall) y tratar el sistema como una **ayuda** a la supervisión humana, nunca como su sustituto ni como base para sanciones automáticas. La decisión de dónde fijar el umbral es, en el fondo, una decisión ética y no meramente técnica.

**Privacidad de los trabajadores.** El sistema procesa imágenes de personas identificables en su lugar de trabajo de forma continua, con riesgo de vigilancia laboral desproporcionada: los mismos flujos podrían reutilizarse para medir productividad, controlar pausas o perfilar individuos. Mitigaciones: minimizar datos (procesar en el borde y **no almacenar** video crudo, solo conteos/eventos agregados), **anonimizar** rostros, limitar la retención, restringir el acceso, ser transparentes con la plantilla y sus representantes, y cumplir la normativa de protección de datos. El propósito legítimo (seguridad) no justifica por sí solo cualquier nivel de recolección.

**Sesgos de los datos de entrenamiento.** El modelo aprende la distribución de sus datos. SHWD combina escenas de obra con multitudes genéricas (cabezas descubiertas); si sobre-representa ciertos tipos de escena, condiciones de luz, colores de casco o complexiones, el rendimiento será desigual entre subgrupos. El fuerte desbalance hace además que el sistema sea, por construcción, sensible al rendimiento en la clase minoritaria. Recomendamos auditar el rendimiento por subgrupos, equilibrar/aumentar la clase minoritaria, documentar las limitaciones del dataset y revalidar el modelo en cada nueva obra antes de confiar en él. La equidad no es una propiedad emergente: debe verificarse explícitamente.

## 7. Discusión Crítica y Limitaciones

- **Resolución de entrada.** Usamos 640 px por restricción de memoria; para la detección a **larga distancia** una resolución mayor (p. ej. 1280) o el *tiling* de la imagen probablemente reducirían la caída de recall en objetos pequeños observada en §5.2. Es la limitación más relevante respecto al objetivo declarado, y se refleja en el bajo AP@0.5:0.95 de la clase *head*.
- **Proxy de oclusión.** El IoU mutuo entre cajas captura solape inter-objeto pero **no** la oclusión por estructuras estáticas (andamios, maquinaria), y está **confundido con el tamaño** del objeto, por lo que no es un estimador fiable de oclusión de forma aislada. Una partición manual o un dataset con etiquetas de oclusión daría una medida más fiel.
- **Desbalance y dominio del dataset.** En SHWD la clase "sin casco" es mayoritaria (por las multitudes), al revés que en una obra real donde el incumplimiento es escaso; las métricas agregadas deben leerse con ese sesgo en mente. Convendría recolección dirigida, sobremuestreo o *focal loss*.
- **Dominio y generalización.** Entrenado y evaluado sobre un único dataset; el rendimiento puede caer en obras, climas o cámaras distintas (*domain shift*). La demostración sobre el test es "no vista" pero proviene de la misma distribución.
- **Verificador basado en reglas.** La clasificación directa es simple y robusta, pero hereda los errores del detector (una cabeza pequeña mal clasificada produce un falso positivo/negativo de cumplimiento) y no incorpora suavizado temporal entre fotogramas de video, que estabilizaría la tasa.
- **Componente propio vs. backbone.** Conforme a la consigna, el backbone es preentrenado; los componentes implementados por el estudiante son el **verificador de cumplimiento**, el **pipeline de evaluación estratificada** (distancia/oclusión), la **conversión VOC→YOLO** y el análisis del dataset.

## 8. Conclusión

Entrenamos y evaluamos críticamente un detector YOLOv8 para el cumplimiento del uso de casco en obras, usando el dataset recomendado SHWD con ejemplos explícitos de "sin casco". Más allá del mAP agregado (**0.942** @0.5), mostramos —mediante un análisis estratificado por distancia y oclusión— que el rendimiento se degrada justamente en los trabajadores lejanos y ocluidos, y construimos un verificador de cumplimiento utilizable cuyo despliegue analizamos desde una perspectiva ética. El trabajo futuro pasa por mayor resolución/tiling, equilibrado de la clase minoritaria y suavizado temporal en video.

## Referencias

Ver `report/related_work.md` para las entradas completas (y `report/neurips/references.bib` en formato BibTeX). Claves: [redmon2016yolo], [bochkovskiy2020yolov4], [wang2024yolov9], [jocher2023yolov8], [fang2018nonhardhat], [hayat2022helmet], [wang2018repulsion], [lin2017fpn], [kang2025sodsurvey], [njvisionpower2019shwd].
