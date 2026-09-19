# Práctica de Big Data  Dask + Spark + Docker

**Institución Universitaria de Envigado** Facultad de Ingenierías **Asignatura:** Big Data
**Docente:** Andrés Felipe Hernández Marulanda

**Integrantes:** Natalia Flores Pérez, Santiago Arcila Gutiérrez, Alejandro Restrepo Uribe y Antonio Patiño Montoya

---

# Qué hace este proyecto

Un pipeline de Big Data reproducible sobre un catálogo astronómico real en el que
Dask y Spark se usan de forma complementaria dentro de un mismo flujo, no como dos
ejercicios sueltos.

```
CSV particionados ── DASK ── Parquet ── SPARK ── resultados agregados ── figuras
  (30 archivos)      ingesta   particionado  agregaciones      CSV pequeños
   3,0 M filas       limpieza   por región   ventana, join
   553 MB            derivadas               SQL, MLlib
```

El acoplamiento es real, la etapa de Spark no lee los CSV originales, lee el Parquet
que escribió Dask, si la etapa de Dask no corre, la de Spark no tiene entrada.

---

# Estructura

```
proyecto-big-data/
├── data/
│   ├── raw/star_classification.csv     catálogo SDSS DR17 real (100.000 objetos)
│   └── interim/                         generado: CSV por época, Parquet
├── notebooks/
│   └── exploracion.ipynb                exploración del catálogo
├── src/
│   ├── config.py                        rutas, escalas y registro de métricas
│   ├── 00_generar_datos.py              amplía el catálogo a N épocas
│   ├── 01_dask_ingesta.py               DASK: ingesta, limpieza, Parquet
│   ├── 02_spark_procesamiento.py        SPARK: agregaciones, ventana, join, SQL, MLlib
│   ├── 03_algoritmo_genetico.py         actividad 3.1
│   ├── 04_caso_medico.py                actividad 3.3
│   ├── 05_comparacion.py                Dask vs Spark, misma operación
│   ├── 06_visualizaciones.py            figuras a partir de agregados
│   └── run_pipeline.py                  orquestador
├── outputs/
│   ├── figuras/                         PNG generados
│   └── metricas/                        CSV de tiempos y resultados
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Ejecución con Docker (la vía recomendada)

```bash
# 1) Construir la imagen
docker build -t practica-bigdata 

# 2) Ejecutar el pipeline completo
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  practica-bigdata
```

Por defecto usa `ESCALA=pequena` (4 épocas, unas 400.000 filas) que corre en pocos
minutos, para la medición completa:

```bash
docker run --rm -e ESCALA=grande -m 4g \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" \
  practica-bigdata
```

# Con docker compose

```bash
docker compose up pipeline                      # pipeline completo
ESCALA=grande docker compose up pipeline        # con el conjunto grande

# Jupyter Lab para explorar (servicio opcional)
docker compose --profile exploracion up jupyter
# luego abrir http://localhost:8888  (sin contraseña)
# la interfaz de Spark queda en http://localhost:4040 mientras haya un job activo
```

---

# Ejecución sin Docker

Requiere Python 3.10 o superior y un **JDK 17** instalado (PySpark lo necesita).

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/run_pipeline.py                  # todo
python src/run_pipeline.py --solo 1 2       # solo las etapas de Dask y Spark
ESCALA=grande python src/run_pipeline.py    # conjunto completo
```

Comprobar que Java está disponible:

```bash
java -version      # debe reportar 17 o superior
```

---

# Escalas disponibles

| `ESCALA` | Épocas | Filas | CSV | Uso |
|---|---|---|---|---|
| `pequena` | 4 | ~0,4 M | ~74 MB | prueba rápida, valor por defecto en Docker |
| `media` | 12 | ~1,2 M | ~221 MB | medición intermedia |
| `grande` | 30 | ~3,0 M | ~553 MB | medición principal del informe |

Se cambia con la variable de entorno, sin tocar el código eso es lo que permite
cumplir el requisito de medir al menos dos configuraciones

---

# Los datos

**Origen real.** `star_classification.csv` es el conjunto *Stellar Classification Dataset
SDSS17*: 100.000 objetos del Sloan Digital Sky Survey Data Release 17, con fotometría en
las cinco bandas (u, g, r, i, z), redshift espectroscópico y clase espectral verificada
(galaxia, estrella o cuásar).

**Ampliación documentada.** La guía admite "construir una versión ampliada o
particionada" el script `00_generar_datos.py` simula que el mismo campo del cielo se
observa en N noches distintas, cada época vuelve a medir los mismos objetos con ruido
fotométrico propio de esa noche y de ese campo, las posiciones y la clase espectral no
cambian, porque son propiedades del objeto y no de la observación.

Esa ampliación no es decorativa es lo que hace que tengan sentido el particionamiento
espacial, las funciones de ventana y la coadición de épocas que se usa antes de entrenar
el modelo.

---

# Actividades de la guía y dónde están resueltas

| Actividad | Dónde |
|---|---|
| 3.1 Algoritmos genéticos, versión secuencial | `src/03_algoritmo_genetico.py` |
| 3.1 Paralelización con Dask y comparación | `src/03_algoritmo_genetico.py` |
| 3.2 Caso astronómico (artículo + reproducción) | `src/01_dask_ingesta.py`, `src/02_spark_procesamiento.py`, informe |
| 3.3 Caso médico (artículo + ejercicio equivalente) | `src/04_caso_medico.py` |
| 3.4 Ingesta y particionamiento con Dask | `src/01_dask_ingesta.py` 1.1 |
| 3.4 Limpieza, 2+ derivadas, Parquet particionado | `src/01_dask_ingesta.py` 1.2–1.4 |
| 3.4 Spark lee la salida de Dask | `src/02_spark_procesamiento.py` 2.1 |
| 3.4 Funciones de ventana | `src/02_spark_procesamiento.py` 2.4 |
| 3.4 Join entre datasets | `src/02_spark_procesamiento.py` 2.5 |
| 3.4 Spark SQL | `src/02_spark_procesamiento.py` 2.6 |
| 3.4 Modelo con MLlib | `src/02_spark_procesamiento.py` 2.7 |
| 3.4 Comparación de tiempos Dask vs Spark | `src/05_comparacion.py` |
| 3.4 Tres o más visualizaciones | `src/06_visualizaciones.py` → `outputs/figuras/` |
| 3.5 Docker y reproducibilidad | `Dockerfile`, `docker-compose.yml`, este README |

---

# Salidas que produce

**outputs/metricas/tiempos.csv** es el registro maestro, cada operación de cada etapa
anota motor, escala, filas, segundos y memoria del proceso, de ahí sale la tabla
comparativa del informe.

Además: comparacion_dask_spark.csv, resumen_por_clase.csv, consulta_sql.csv,
importancia_mllib.csv, matriz_confusion_mllib.csv, ag_comparacion.csv,
ag_escalabilidad.csv, medico_importancia.csv y los agregados para graficar.

**outputs/figuras/** contiene siete PNG, todos generados a partir de los CSV agregados,
en ningún momento se convierte el conjunto completo a pandas para graficar.

---

# Notas sobre el entorno de medición

Los tiempos del informe se tomaron en una máquina con **1 núcleo y 4 GB de RAM**, eso
condiciona dos resultados que conviene leer con ese contexto:

- La paralelización del algoritmo genético **no acelera nada**, porque el techo de
  aceleración es el número de núcleos, el código está correctamente paralelizado; el
  entorno no permite aprovecharlo, está medido y discutido en el informe.
- Spark corre en modo local[*] es decir, un solo proceso simulando un clúster, sus
  ventajas de distribución real no se ven, lo que sí se ve es su motor de ejecución.
