# Práctica de Big Data Dask, Apache Spark y Docker

**Integrantes:** Natalia Flores Pérez, Santiago Arcila Gutiérrez, Alejandro Restrepo Uribe y Antonio Patiño Montoya


## 1. Resumen

Construimos un flujo de Big Data reproducible sobre un catálogo astronómico real del
Sloan Digital Sky Survey, en el que Dask y Spark se usan de forma complementaria dentro
de un mismo pipeline, Dask se encarga de la ingesta particionada, la limpieza y la
generación de variables derivadas, y entrega su salida en Parquet particionado por región
del cielo. Spark toma ese Parquet y ejecuta las operaciones que requieren redistribución
de datos agregaciones, funciones de ventana, un join entre dos tablas, consultas SQL y
un modelo de clasificación con MLlib

El acoplamiento entre los dos motores es real y verificable, la etapa de Spark no lee los
archivos originales, sino la salida de Dask si la etapa de Dask no se ejecuta, la de
Spark no tiene entrada

Todo el proyecto se empaqueta con Docker y corre con un solo comando.


# 2. Entorno de medición

| Recurso | Valor |
|---|---|
| Núcleos de CPU | 1 |
| Memoria RAM | 4 GB |
| Python | 3.12 |
| Dask | 2026.8.0 |
| PySpark | 3.5.3 (modo `local[*]`) |
| Java | OpenJDK 21 |
| Memoria del driver de Spark | 2 GB |
| Particiones de shuffle | 8 |

Este entorno condiciona dos resultados del informe y conviene tenerlo presente al leerlos
con un solo núcleo, ninguna paralelización puede acelerar nada y Spark corre como un
único proceso que simula un clúster.


# 3. Actividad 3.1 Algoritmos genéticos y procesamiento paralelo

# 3.1.1 Qué son

Un algoritmo genético es una técnica de optimización inspirada en la evolución biológica,
en lugar de buscar el óptimo con cálculo diferencial, mantiene una población de soluciones
candidatas que compiten y se reproducen durante generaciones sucesivas.

**Representación.** Cada individuo es un vector binario de longitud N: un 1 significa
"incluyo este objeto" y un 0 "lo dejo fuera".

**Función de aptitud.** Es lo único que el algoritmo sabe del problema; todo lo demás es
maquinaria genérica. Aquí es el valor científico total de los objetos escogidos, con
penalización proporcional si se excede la capacidad.

**Selección.** Por torneo: se toman k individuos al azar y gana el de mejor aptitud.
Mantiene presión selectiva sin que los mejores dominen de inmediato.

**Cruce.** Uniforme: cada gen del hijo se toma al azar de uno u otro padre.

**Mutación.** Cambio aleatorio de algunos genes. Es lo que impide que la población se
estanque: sin mutación, si todos los individuos convergen al mismo, el cruce ya no genera
nada nuevo.

**Ventajas.** No necesitan derivadas ni que la función objetivo sea continua o convexa.
Exploran varias regiones a la vez, lo que reduce el riesgo de quedarse en un óptimo local.
Se adaptan a problemas combinatorios. Y la evaluación de la población es
*vergonzosamente paralela*: cada individuo se evalúa sin depender de los demás.

**Desventajas.** No garantizan el óptimo global. Tienen muchos hiperparámetros y el
resultado depende bastante de ellos. Son caros en número de evaluaciones. Y no son
reproducibles sin fijar la semilla.

### 3.1.2 El problema planteado

Programación de observaciones de un telescopio: una noche con tiempo limitado, y una lista
de objetos candidatos, cada uno con un valor científico y un costo en minutos. Hay que
elegir el subconjunto que maximice el valor sin pasarse del tiempo disponible.

Es el problema de la mochila 0/1, que es NP-duro. Con 250 objetos hay 2²⁵⁰ combinaciones
posibles, más que átomos en el universo observable. Por eso se usa una heurística.

### 3.1.3 Resultados

Parámetros: 250 objetos, población de 400, 400 generaciones, 8 semillas independientes.

| Versión | Tiempo | Mejor valor | Aceleración |
|---|---|---|---|
| Secuencial | 13,14 s | 7.331,6 | 1,00× |
| Dask (procesos) | 13,43 s | 7.331,6 | 0,98× |
| Dask (hilos) | 12,88 s | 7.331,6 | 1,02× |

Escalabilidad al variar el número de trabajadores de Dask:

| Trabajadores | Tiempo | Aceleración |
|---|---|---|
| 1 | 13,46 s | 0,98× |
| 2 | 14,00 s | 0,94× |
| 4 | 13,96 s | 0,94× |
| 8 | 13,82 s | 0,95× |

### 3.1.4 ¿Aporta la paralelización?

**En este entorno, no. Y la razón es el entorno, no el código.**

El techo teórico de aceleración es exactamente el número de núcleos disponibles. Esta
máquina tiene uno, así que repartir el trabajo entre 2, 4 u 8 procesos no puede hacer nada
más que añadir el costo de arrancarlos y de serializar datos entre ellos. Eso explica que
todas las configuraciones queden ligeramente por debajo de 1×: es la misma cantidad de
cálculo más un sobrecosto.

Con el planificador de hilos tampoco hay ganancia, pero por una razón distinta: el GIL de
Python impide que dos hilos ejecuten bytecode a la vez. No sale peor que el secuencial
porque NumPy libera el GIL durante sus operaciones internas, que es donde está casi todo
el cálculo.

La conclusión honesta es que el código está correctamente paralelizado y el problema es
paralelizable por naturaleza, pero el hardware no permite aprovecharlo. En una máquina de
8 núcleos el mismo código, sin cambios, mostraría ganancia real.

**Un hallazgo del desarrollo que vale la pena reportar.** La primera versión del algoritmo
quedaba un 1,9% **por debajo** de una simple heurística voraz que ordena por relación
valor/costo. La causa era que el algoritmo gastaba la mayoría de las generaciones
aprendiendo a no excederse de la capacidad en lugar de aprender qué objetos convenía
llevar. Al añadir un **operador de reparación**, que vuelve factible a todo individuo
quitándole primero los objetos de peor relación valor/costo, el algoritmo pasó a superar
a la heurística. Es un recordatorio de que en algoritmos genéticos el diseño de los
operadores pesa más que el ajuste de hiperparámetros.

## 4. Actividad 3.2 · Caso científico: datos astronómicos

### 4.1 Artículo de referencia

Caplar, N., Beebe, W., Branton, D., Campos, S., Connolly, A., DeLucchi, M., Jones, D.,
Jurić, M., Kubica, J., Malanchev, K., Mandelbaum, R. y McGuire, S. (2025).
*Using LSDB to enable large-scale catalog distribution, cross-matching, and analytics.*
arXiv:2501.02103.

Referencias complementarias:

- Le Boulc'h, Q. et al. (2024). *The Rubin Observatory's Legacy Survey of Space and Time
  DP0.2 processing campaign at CC-IN2P3.* arXiv:2404.06234.
- NSF–DOE Vera C. Rubin Observatory. Cifras oficiales de volumen de datos del LSST.

### 4.2 Análisis del artículo

**Problema.** El Observatorio Vera C. Rubin producirá volúmenes de datos sin precedentes
en astronomía óptica. Distribuir esos catálogos y permitir que un investigador haga
análisis de cielo completo sobre ellos es un problema abierto.

**Fuente y volumen.** El LSST generará alrededor de 20 TB de imágenes crudas por noche y
unos 60 PB en los diez años del sondeo. El catálogo final tendrá del orden de 20 mil
millones de galaxias y 17 mil millones de estrellas, con aproximadamente 30 billones de
fuentes observadas. Los catálogos que el artículo maneja ya superan los 10 TB.

**Tecnologías.** El equipo del LINCC Frameworks desarrolló dos piezas: **HATS**
(*Hierarchical Adaptive Tiling Scheme*), un formato que particiona el cielo con píxeles
HEALPix de orden variable para que todas las particiones tengan un tamaño similar, y
**LSDB**, el paquete de análisis que trabaja sobre ese formato. HATS usa **Apache Parquet**
como almacenamiento subyacente, y **LSDB usa Dask para la paralelización**.

**Metodología.** El particionamiento se hace dividiendo iterativamente las regiones del
cielo hasta que cada partición quede por debajo de un umbral de tamaño. Al ser Parquet
columnar, si el usuario solo necesita unas columnas se leen únicamente esas, lo que reduce
la entrada/salida y el uso de memoria.

**Resultados.** Demostraron el funcionamiento sobre catálogos de ZTF y Pan-STARRS, en
clúster y en nube. Entre otros resultados, extrajeron un periodograma de baja resolución
para mil millones de curvas de luz de ZTF DR14 en dos horas usando siete nodos del
supercomputador Bridges-2.

### 4.3 Qué reprodujimos y qué adaptamos

Este artículo encaja de forma casi literal con lo que pide la guía: usa Dask, usa Parquet
y usa particionamiento espacial. No pudimos reproducirlo tal cual porque los catálogos
HATS pesan decenas de terabytes y el procesamiento se hizo en un supercomputador.

**Lo que sí reprodujimos** es el esquema completo a escala reducida, con datos reales:

| Elemento del artículo | Nuestra implementación |
|---|---|
| Catálogo astronómico real | SDSS DR17, 100.000 objetos con fotometría de 5 bandas |
| Particionamiento espacial jerárquico (HEALPix) | Celdas de 30°×30° en ascensión recta y declinación |
| Almacenamiento en Parquet | Parquet con compresión snappy, particionado por región |
| Paralelización con Dask | Dask DataFrame con 30 particiones |
| Análisis de catálogo a gran escala | Agregaciones, ventana, join y MLlib en Spark |

**Ampliación del conjunto.** El catálogo real son 100.000 filas, insuficientes para hablar
de Big Data. La guía admite explícitamente construir una versión ampliada o particionada.
Simulamos que el mismo campo del cielo se observa en 30 noches distintas: cada época
vuelve a medir los mismos objetos con ruido fotométrico dependiente de la magnitud del
objeto, del seeing de esa noche y de la calidad intrínseca de ese campo. Las posiciones y
la clase espectral no cambian, porque son propiedades del objeto y no de la observación.

El resultado son **3.000.000 de filas en 30 archivos CSV, 553 MB**. La ampliación no es
decorativa: es lo que hace que tengan sentido el particionamiento espacial, las funciones
de ventana por región y la coadición de épocas que describimos más adelante.

### 4.4 Resultados del análisis

Agregación por clase espectral sobre 2.928.603 observaciones válidas:

| Clase | Observaciones | Redshift medio | Color g−r | Color r−i | Magnitud media |
|---|---|---|---|---|---|
| GALAXY | 1.741.045 | 0,4216 | 1,3185 | 0,7356 | 20,08 |
| STAR | 632.410 | −0,0001 | 0,6703 | 0,4031 | 19,32 |
| QSO | 555.148 | 1,7198 | 0,3022 | 0,1930 | 20,76 |

**La física cuadra**, y eso es una validación del pipeline: las estrellas de nuestra propia
galaxia tienen redshift esencialmente cero, las galaxias están a distancias intermedias y
los cuásares son los objetos más lejanos, con redshift medio de 1,72, lo que corresponde a
unos 7.400 megaparsecs. Los cuásares son además los más azules (color g−r de 0,30), como
se espera de su emisión no térmica.

El diagrama color-color (figura 2) muestra la secuencia estelar bien definida para las
estrellas, la agrupación azul de los cuásares y la bimodalidad característica de las
galaxias entre la secuencia roja y la nube azul. Ninguna de esas estructuras fue impuesta:
emergen de los datos reales.

## 5. Actividad 3.3 · Caso científico: datos médicos

### 5.1 Artículo de referencia

Chen, S. et al. (2025). *Machine Learning-Based Prediction of ICU Mortality in
Sepsis-Associated Acute Kidney Injury Patients Using MIMIC-IV Database with Validation
from eICU Database.* arXiv:2502.17978.

Bases de datos citadas:

- Johnson, A. E. W. et al. (2023). *MIMIC-IV, a freely accessible electronic health record
  dataset.* Scientific Data 10, 1.
- Pollard, T. J. et al. (2018). *The eICU Collaborative Research Database.*
  Scientific Data 5.

### 5.2 Análisis del artículo

**Problema.** La lesión renal aguda asociada a sepsis (SA-AKI) tiene mortalidad alta en
cuidados intensivos. Identificar temprano a los pacientes de mayor riesgo permitiría
intervenir antes.

**Fuente y volumen.** MIMIC-IV reúne los registros clínicos de todos los ingresos al Beth
Israel Deaconess Medical Center de Boston entre 2008 y 2022, con signos vitales,
laboratorios, medicación y desenlaces. El estudio identificó 9.474 pacientes con SA-AKI.

**Arquitectura y técnicas.** Selección de variables con factor de inflación de la varianza
y eliminación recursiva de características, reduciendo a 24 predictoras; modelo XGBoost
con búsqueda en rejilla; interpretabilidad con SHAP y LIME; validación externa en eICU.

**Resultados.** El modelo alcanza buena discriminación, y la interpretabilidad señala el
lactato sérico, la puntuación APACHE II, la diuresis total y el calcio sérico como las
variables más influyentes.

### 5.3 Por qué no se reproduce y cómo se adaptó

MIMIC-IV no es de descarga libre: exige credencial de PhysioNet, formación certificada en
investigación con sujetos humanos y firma de un acuerdo de uso de datos. La guía advierte
además que hay que respetar la privacidad y usar datos abiertos, anonimizados o sintéticos.

Construimos una **cohorte sintética** de 400.000 estancias en UCI que reproduce la
*estructura* del estudio, no sus datos. Ningún registro corresponde a una persona real. Las
variables se generan con distribuciones típicas de cuidados intensivos y las relaciones
con el desenlace se fijan siguiendo lo que reporta la literatura clínica: lactato alto,
APACHE II alto y diuresis baja aumentan el riesgo.

El preprocesamiento usa Dask para ingesta particionada (8 archivos), imputación por
mediana calculada de forma distribuida, cuatro variables derivadas con sentido clínico
(diuresis por kilo-hora, índice de shock, grupo etario, marcador de lactato alto) y
escritura en Parquet.

### 5.4 Resultados

Mortalidad según lactato sérico, sobre la cohorte completa:

| Lactato (mmol/L) | Mortalidad | Estancias |
|---|---|---|
| 2 | 13,6 % | 75.969 |
| 4 | 19,3 % | 15.638 |
| 6 | 24,5 % | 3.538 |
| 8 | 29,9 % | 991 |
| 10 | 38,0 % | 342 |
| 12 | 56,6 % | 304 |

El gradiente es muy marcado: la mortalidad se multiplica por más de cuatro entre el primer
y el último tramo.

Modelo de potenciación de gradiente sobre 80.000 estancias de entrenamiento: **AUC de
0,687** en el conjunto de prueba. Variables más influyentes por importancia de
permutación: APACHE II, creatinina, SOFA y lactato.

**Contraste con el artículo.** El artículo reporta lactato, APACHE II, diuresis y calcio.
Nuestro ranking coincide en gran parte, pero esa coincidencia **no es un hallazgo**: las
relaciones se fijaron al generar los datos siguiendo esa misma literatura. Lo que el
ejercicio valida es el flujo metodológico completo, no una conclusión clínica.

Conviene además notar que el AUC de 0,687 es modesto. Es consistente con haber introducido
ruido deliberado en la generación: un modelo que alcanzara 0,99 sobre datos sintéticos
solo estaría demostrando que memorizó la fórmula con la que se crearon.


## 6. Actividad 3.4 · Pipeline integrador Dask + Spark

### 6.1 Arquitectura

```
30 CSV (553 MB)
      │
      
┌─────────────────────────────────────────┐
│ DASK                                    │
│  · read_csv con 30 particiones          │
│  · centinelas −9999 → nulo              │
│  · filtros de rango físico              │
│  · 4 variables derivadas                │
│  · tabla secundaria de campos           │
└─────────────────────────────────────────┘
      │  to_parquet(partition_on="region")
      
  Parquet particionado (414 MB)  ── formato de intercambio
      │
      
┌─────────────────────────────────────────┐
│ SPARK                                   │
│  · read.parquet                         │
│  · filtros y transformaciones           │
│  · groupBy + agregados                  │
│  · Window: row_number, percent_rank     │
│  · broadcast join con campos            │
│  · Spark SQL                            │
│  · MLlib RandomForest                   │
└─────────────────────────────────────────┘
      │
      
  CSV agregados pequeños  ──  7 figuras
```

### 6.2 Justificación de qué motor hace qué

**Dask para la ingesta, limpieza y transformación.** Estas operaciones se aplican de forma
independiente a cada partición y no requieren mover datos entre nodos. Tres razones
concretas para elegir Dask aquí:

1. *Costo de arranque.* Medido: Dask queda listo en 0,44 s; Spark necesita 15,03 s para
   levantar la JVM. En una etapa de preprocesamiento que se ejecuta una vez, ese costo
   fijo pesa.
2. *Integración con el ecosistema científico.* El código de limpieza es esencialmente
   pandas y NumPy, y en Dask se reutiliza tal cual. En Spark habría que reescribirlo con
   la API de DataFrames o pagar el sobrecosto de las UDF de Python, que serializan datos
   entre la JVM y el intérprete.
3. *Control del particionamiento de salida.* Escribimos Parquet particionado por región
   del cielo, replicando el esquema de HATS.

**Spark para agregaciones, ventana, join, SQL y modelado.** Estas operaciones requieren
*shuffle*, es decir, redistribuir datos según una clave. Spark está construido alrededor
de ese problema: su optimizador Catalyst reordena las operaciones antes de ejecutarlas,
elige la estrategia de join según el tamaño de las tablas y gestiona el desbordamiento a
disco. Además ofrece SQL sobre datos distribuidos y MLlib, que Dask no tiene de forma
nativa.

Los tiempos medidos respaldan esa división: en las tres operaciones con shuffle Spark
resultó entre 2,1× y 6,1× más rápido que Dask.

### 6.3 Por qué Parquet como formato de intercambio

Cuatro razones, y las tres primeras se traducen en tiempo:

- **Es columnar.** Spark lee solo las columnas que necesita. En la comparación de la
  sección 7 esto resultó decisivo: cuando dejamos que Dask leyera las 20 columnas mientras
  Spark leía 7, la diferencia de tiempos se inflaba artificialmente.
- **Está comprimido.** De 553 MB en CSV a 414 MB en Parquet, un 25% menos de disco que
  leer.
- **Guarda el esquema.** Spark no tiene que inferir tipos, que es una fuente clásica de
  errores cuando una columna se ve entera al principio del archivo y trae nulos después.
- **Es el formato del artículo de referencia.** HATS usa Parquet por exactamente estos
  motivos.

### 6.3.1 Un hallazgo del desarrollo: el problema de los archivos pequeños

Al medir la lectura nos encontramos con lo contrario de lo esperado: **el CSV resultaba
más rápido que el Parquet** para un escaneo completo. La causa no era el formato sino cómo
lo escribimos.

Particionar por región genera un archivo por cada combinación de región y partición de
entrada: 45 regiones × 30 particiones = **1.350 archivos de unos 108 KB de mediana**. Cada
archivo tiene un costo fijo de apertura y de lectura de metadatos, así que un escaneo
completo paga 1.350 veces ese costo.

Es el llamado *problema de los archivos pequeños*, bien conocido en sistemas de datos
distribuidos. La solución fue escribir además una versión compacta, reparticionada en 8
archivos grandes. Medición del mismo cálculo (media del redshift sobre 3 millones de filas):

| Origen | Archivos | Tiempo |
|---|---|---|
| CSV | 30 | 2,46 s |
| Parquet particionado por región | 1.350 | 3,22 s |
| Parquet compacto | 8 | **0,05 s** |

El Parquet compacto es **70 veces más rápido** que el particionado para un escaneo completo.

**Pero eso no significa que el particionamiento esté mal.** Sigue siendo la opción correcta
para consultas *filtradas por región*, porque el motor puede saltarse los archivos que no
necesita (*partition pruning*), que es exactamente el caso de uso del artículo de
referencia: un astrónomo que quiere analizar una zona del cielo, no el catálogo entero.

La conclusión práctica es que no hay un esquema mejor en abstracto. Depende de cómo se
vayan a consultar los datos, y en un sistema real conviene mantener ambas
representaciones. El pipeline escribe las dos.

### 6.4 Resultados de la etapa de Spark

Etapa completa en **113,1 segundos** sobre 3 millones de filas, incluyendo el arranque de
la JVM.

| Operación | Tiempo |
|---|---|
| Lectura del Parquet de Dask | 11,4 s |
| Filtros y transformaciones | 6,9 s |
| Agregación por clase | 9,6 s |
| Función de ventana (top 10 por región) | 10,0 s |
| Broadcast join con 856 campos | 7,4 s |
| Spark SQL (19 grupos) | 6,3 s |
| MLlib RandomForest | 28,2 s |

**Resultado del join.** Al cruzar el catálogo con la tabla de campos observacionales se ve
el efecto de la calidad de la observación sobre la fotometría:

| Calidad del campo | Observaciones | Seeing medio | Dispersión del color g−r |
|---|---|---|---|
| Buena | 1.978.788 | 1,024″ | 0,7320 |
| Media | 797.976 | 1,477″ | 0,7451 |
| Pobre | 151.839 | 2,037″ | 0,7459 |

La dispersión del color crece cuando el seeing empeora, que es exactamente lo que se
espera físicamente: peor seeing significa medidas más ruidosas.

### 6.5 El modelo de MLlib y un problema metodológico que hubo que resolver

**El problema.** Cada objeto del cielo fue observado en las 30 épocas, así que aparece 30
veces en la tabla. Una partición aleatoria de las filas metería el mismo objeto en
entrenamiento y en prueba a la vez. El modelo reconocería un objeto que ya vio y la
exactitud saldría inflada. Es fuga de información, y es sutil porque las filas sí son
distintas entre sí: cambian por el ruido fotométrico.

**La solución.** La misma que usan los sondeos reales: construir un catálogo
**coadicionado**, promediando las épocas de cada objeto para obtener una medición más
profunda y con menos ruido. Eso deja una fila por objeto y elimina la fuga de raíz. Además
es otra agregación distribuida, que es donde Spark rinde.

De 2.928.603 observaciones se obtuvieron **83.481 objetos únicos**.

**Resultados.** RandomForest con 40 árboles y profundidad 8:

- Exactitud: **0,9689**
- F1: **0,9688**

Importancia de variables:

| Variable | Importancia |
|---|---|
| redshift | 0,6977 |
| color_gr | 0,1155 |
| color_ri | 0,1104 |
| color_ug | 0,0327 |
| color_iz | 0,0234 |
| magnitud_media | 0,0203 |

El redshift domina, lo cual tiene sentido físico directo: las estrellas están a redshift
cero por construcción y los cuásares a redshift alto, así que esa sola variable ya separa
dos de las tres clases. Los índices de color, que son las variables derivadas que creamos
en la etapa de Dask, aportan en conjunto un 28% de la importancia y son los que permiten
distinguir galaxias de cuásares en el rango de redshift donde se solapan.

## 7. Comparación Dask vs Spark

### 7.1 Condiciones de la medición

Una comparación mal montada da números que no significan nada. Estas son las precauciones
que tomamos:

1. **Los dos motores leen el mismo Parquet y exactamente las mismas 7 columnas.** En una
   primera versión Dask leía las 20 columnas mientras Spark, gracias a su optimizador, solo
   leía las necesarias. Eso arrojaba factores de hasta 20× a favor de Spark que eran un
   artefacto de la medición, no una propiedad del motor. Corregido, los factores bajaron a
   entre 2,1× y 6,1×.
2. **Los datos quedan materializados en memoria en ambos lados** antes de medir: `.cache()`
   en Spark y `.persist()` en Dask.
3. **El arranque de la JVM se mide aparte** y no se suma a las operaciones, porque es un
   costo fijo por sesión.
4. **Se descarta una ejecución de calentamiento** y se reporta el mejor tiempo de las
   siguientes.

### 7.2 Tabla comparativa

Arranque de los motores:

| Motor | Arranque |
|---|---|
| Spark | 15,03 s |
| Dask | 0,44 s |

Operaciones (sin incluir el arranque):

| Operación | Tamaño | Filas | Dask | Spark | Ganador | Factor |
|---|---|---|---|---|---|---|
| A. Agregación simple (groupBy 1 clave) | 50 % | 1.465.359 | 2,45 s | 1,18 s | Spark | 2,08× |
| B. Agregación compuesta (2 claves) | 50 % | 1.465.359 | 4,42 s | 0,99 s | Spark | 4,47× |
| C. Join + agregación | 50 % | 1.465.359 | 3,39 s | 1,01 s | Spark | 3,37× |
| A. Agregación simple (groupBy 1 clave) | 100 % | 2.928.603 | 2,58 s | 0,83 s | Spark | 3,12× |
| B. Agregación compuesta (2 claves) | 100 % | 2.928.603 | 4,52 s | 0,75 s | Spark | 6,05× |
| C. Join + agregación | 100 % | 2.928.603 | 3,58 s | 0,80 s | Spark | 4,47× |

### 7.3 Discusión

**Spark gana en todas las operaciones con shuffle, y su ventaja crece con el tamaño.** En
la agregación compuesta pasa de 4,47× a 6,05× al duplicar los datos. La explicación es que
Catalyst optimiza el plan antes de ejecutarlo: reordena filtros para aplicarlos lo antes
posible y elige la estrategia de agregación. El planificador de Dask es más simple y
ejecuta más cerca de lo que el usuario escribió.

**Un detalle que llama la atención: los tiempos de Spark bajan al duplicar los datos.** En
la operación B pasa de 0,99 s a 0,75 s con el doble de filas. No es magia: al 50% se aplica
un muestreo aleatorio que obliga a leer todo el archivo para descartar la mitad, mientras
que al 100% ese paso desaparece. Es un recordatorio de que un muestreo no es gratis.

**Dask no queda mal parado donde le corresponde.** Su arranque es 34 veces más rápido, y
en la etapa de ingesta y limpieza, que no aparece en esta tabla porque no tiene equivalente
directo, procesó 3 millones de filas y escribió el Parquet en 16,5 segundos con 280 MB de
memoria de proceso.

**La lectura práctica** es que la comparación "cuál es más rápido" está mal planteada. La
pregunta correcta es qué operación se va a ejecutar. Para preprocesar archivos y escribir
resultados, Dask. Para agregar, unir y consultar, Spark. Que es exactamente la división
que adoptamos en el pipeline.

## 8. Actividad 3.5 · Docker y reproducibilidad

El proyecto incluye `Dockerfile`, `docker-compose.yml` y `requirements.txt` con versiones
fijadas.

La imagen parte de `python:3.11-slim` e instala OpenJDK 17, que PySpark necesita. El
`requirements.txt` se copia antes que el código para que Docker reutilice la capa de
dependencias cuando solo cambia el código, lo que ahorra minutos en cada reconstrucción.

```bash
docker build -t practica-bigdata .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" practica-bigdata
```

Por defecto el contenedor usa `ESCALA=pequena` (unas 400.000 filas) para que la ejecución
de prueba termine en pocos minutos. Con `-e ESCALA=grande` corre el conjunto completo.

El `docker-compose.yml` define dos servicios: `pipeline`, que ejecuta el flujo y termina,
y `jupyter`, un servicio opcional bajo perfil que levanta Jupyter Lab con el mismo entorno
y expone también el puerto 4040 de la interfaz web de Spark.

## 9. Conclusión técnica: qué herramienta para cada etapa en un escenario real

| Etapa | Herramienta | Por qué |
|---|---|---|
| Ingesta de archivos crudos | Dask | Arranque inmediato, API de pandas, control del particionamiento de salida |
| Limpieza y variables derivadas | Dask | Operaciones fila a fila sin shuffle; el código científico se reutiliza tal cual |
| Formato de intercambio | Parquet | Columnar, comprimido, con esquema, estándar de la industria |
| Agregaciones y joins | Spark | Catalyst optimiza el plan; gestiona shuffle y desbordamiento a disco |
| Consultas ad hoc | Spark SQL | Permite que analistas sin Python consulten los mismos datos |
| Modelado a escala | Spark MLlib | Entrena de forma distribuida sin sacar los datos del clúster |
| Visualización | pandas + matplotlib | Sobre resultados ya agregados, que son pequeños por definición |

**Una advertencia sobre el umbral.** Nada de esto aplica por debajo de cierto tamaño. Con
datos que caben cómodamente en memoria, pandas es más rápido que ambos y mucho más simple.
La complejidad de un motor distribuido solo se justifica cuando el archivo no cabe o cuando
el cálculo tarda lo suficiente como para que valga la pena repartirlo.

## 10. Respuestas a las preguntas de sustentación

**¿Qué parte del pipeline gana realmente al usar Dask y cuál al usar Spark?**
Dask gana en la ingesta y la limpieza: arranca en 0,44 s frente a los 15 s de Spark, y el
código de limpieza es pandas reutilizado sin cambios. Spark gana en todo lo que requiere
shuffle: medimos entre 2,1× y 6,1× de ventaja en agregaciones y joins, y además aporta SQL
y MLlib, que Dask no tiene de forma nativa.

**¿Qué ocurre si el conjunto crece diez veces? ¿Qué componente se convierte primero en
cuello de botella?**
A 30 millones de filas (unos 5,5 GB en CSV) el primer cuello sería la **memoria del driver
de Spark** en operaciones que colectan resultados, y la **escritura del Parquet en Dask**,
que actualmente tarda 16,5 s y escalaría de forma aproximadamente lineal. La lectura de
disco pasaría a dominar sobre el cálculo. La mitigación inmediata es aumentar el número de
particiones y de `spark.sql.shuffle.partitions`; la de fondo es pasar de `local[*]` a un
clúster real, porque con un solo núcleo no hay paralelismo que explotar.

**¿Por qué eligieron Parquet?**
Porque es columnar (Spark lee solo las columnas que necesita), comprimido (25% menos disco
que CSV en nuestro caso), conserva el esquema y es el formato que usa HATS en el artículo
de referencia. La alternativa natural sería ORC, que es equivalente, o Delta Lake si se
necesitaran transacciones y control de versiones.

**¿Qué diferencia hay entre la evaluación perezosa de Dask y las transformaciones y
acciones de Spark?**
Son el mismo principio con implementaciones distintas. En ambos, las operaciones construyen
un grafo y nada se ejecuta hasta que algo lo fuerza: `.compute()` en Dask, una *acción*
como `count()` o `collect()` en Spark. La diferencia práctica está en lo que ocurre en el
medio: Spark pasa el grafo por Catalyst, que lo reescribe (empuja filtros hacia la lectura,
poda columnas, elige la estrategia de join) antes de ejecutarlo. Dask ejecuta un grafo más
cercano a lo que el usuario escribió. Eso explica buena parte de la diferencia de tiempos
que medimos. Lo comprobamos en el código: `dd.read_csv` sobre 553 MB devuelve en 0,016 s
porque solo construye el plan, y el `len()` posterior tarda 4,6 s, que es cuando lee de
verdad.

**¿Qué problema resuelve Docker en este proyecto y qué problema no resuelve?**
Resuelve la reproducibilidad del *entorno*: fija la versión de Python, de Java, de Spark y
de cada librería, de modo que el pipeline corre igual en cualquier máquina sin que nadie
tenga que instalar un JDK a mano. **No resuelve** el rendimiento: un contenedor no añade
núcleos ni memoria, y de hecho introduce una pequeña sobrecarga. Tampoco resuelve la
distribución real: para eso haría falta un orquestador como Kubernetes o un gestor de
clúster como YARN. Y no resuelve la reproducibilidad de los *datos*, que dependen de una
fuente externa y de la semilla que fijamos.

**¿Cómo modificarían la solución para ejecutarla en un clúster real?**
Tres cambios. Primero, sustituir `.master("local[*]")` por la dirección del gestor de
clúster (`yarn`, `k8s://...` o `spark://...`) y ajustar `spark.executor.instances`,
`executor.cores` y `executor.memory`. Segundo, mover los datos de disco local a un
almacenamiento compartido y accesible por todos los nodos, típicamente HDFS o
almacenamiento de objetos tipo S3, cambiando las rutas por URIs. Tercero, en Dask,
reemplazar el planificador local por un `dask.distributed.Client` apuntando a un
planificador remoto. El código de análisis en sí no cambiaría: esa es precisamente la
ventaja de haber usado las API de alto nivel de ambos motores.

## 11. Limitaciones declaradas

1. **El entorno de medición tiene un solo núcleo.** Ninguna medición de paralelismo de
   este informe puede mostrar aceleración, y los resultados del algoritmo genético hay que
   leerlos con ese contexto. Las comparaciones Dask-Spark siguen siendo válidas porque
   ambos motores sufren la misma limitación.

2. **El conjunto es ampliado, no observado.** Las 3 millones de filas provienen de 100.000
   objetos reales replicados en 30 épocas sintéticas. Las propiedades físicas de los
   objetos son reales; la variación entre épocas es simulada.

3. **La cohorte médica es completamente sintética.** No reproduce un hallazgo clínico, solo
   un flujo metodológico. La coincidencia de las variables importantes con el artículo era
   esperable porque las relaciones se fijaron al generar los datos.

4. **Spark corre en modo local.** No se midió distribución real entre nodos, solo el motor
   de ejecución. Los tiempos no son extrapolables a un clúster sin verificación.

5. **El Parquet particionado genera 1.350 archivos pequeños**, lo que penaliza los
   escaneos completos. Se mitigó escribiendo también una versión compacta, pero un diseño
   de producción ajustaría el número de particiones al volumen real en lugar de escribir
   dos copias.

6. **La aproximación de distancia usa la ley de Hubble simple**, que no es válida para
   redshift alto. Los valores de distancia de los cuásares son ilustrativos, no
   cosmológicamente correctos.

## 12. Referencias

Caplar, N. et al. (2025). *Using LSDB to enable large-scale catalog distribution,
cross-matching, and analytics.* arXiv:2501.02103.

Chen, S. et al. (2025). *Machine Learning-Based Prediction of ICU Mortality in
Sepsis-Associated Acute Kidney Injury Patients Using MIMIC-IV Database with Validation
from eICU Database.* arXiv:2502.17978.

Johnson, A. E. W. et al. (2023). *MIMIC-IV, a freely accessible electronic health record
dataset.* Scientific Data 10, 1.

Le Boulc'h, Q. et al. (2024). *The Rubin Observatory's Legacy Survey of Space and Time
DP0.2 processing campaign at CC-IN2P3.* arXiv:2404.06234.

Pollard, T. J. et al. (2018). *The eICU Collaborative Research Database, a freely available
multi-center database for critical care research.* Scientific Data 5.

Sloan Digital Sky Survey, Data Release 17. Conjunto *Stellar Classification Dataset SDSS17*.

Apache Spark. Documentación oficial. https://spark.apache.org/docs/latest/
Dask. Documentación oficial. https://docs.dask.org/
Apache Parquet. https://parquet.apache.org/
Docker. https://docs.docker.com/
