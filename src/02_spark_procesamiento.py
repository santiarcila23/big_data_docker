"""
Etapa 2 - Procesamiento y analisis avanzado con Spark
=====================================================

AQUI ESTA EL INTERCAMBIO DE DATOS QUE EXIGE LA GUIA: este script NO vuelve a
leer los CSV originales. Lee el Parquet particionado que produjo Dask en la
etapa anterior. Si esa etapa no corrio, este script no tiene entrada. Ese es el
acoplamiento entre los dos motores dentro de un mismo flujo.

Por que Spark en esta etapa y no Dask
-------------------------------------
Lo que viene ahora es distinto de la limpieza: agregaciones por grupo, funciones
de ventana, un join entre dos tablas y consultas SQL. Todas esas operaciones
requieren shuffle, es decir, redistribuir los datos entre particiones segun una
clave.

Spark esta construido justamente alrededor de ese problema. Su motor Catalyst
reordena las operaciones antes de ejecutarlas, decide la estrategia de join
(difusion o por particion) segun el tamano de las tablas, y gestiona el
desbordamiento a disco cuando un grupo no cabe en memoria. Dask puede hacer
shuffles, pero su planificador es mas simple y en joins grandes o ventanas
particionadas se degrada antes.

Ademas, dos capacidades que Dask no ofrece de forma nativa y que la guia pide:
SQL sobre datos distribuidos y una libreria de aprendizaje automatico
distribuida (MLlib).
"""
import time
import pandas as pd

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

from config import (DIR_PARQUET, DIR_CAMPOS, DIR_METRICAS, ESCALA, SEMILLA,
                    SPARK_MEMORIA_DRIVER, SPARK_PARTICIONES,
                    registrar_metrica, medir_memoria)


def crear_sesion():
    return (SparkSession.builder
            .appName("PipelineAstronomico")
            .master("local[*]")
            .config("spark.driver.memory", SPARK_MEMORIA_DRIVER)
            .config("spark.sql.shuffle.partitions", str(SPARK_PARTICIONES))
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def main():
    print(f"[02] SPARK - procesamiento distribuido (escala '{ESCALA}')")
    t_inicio = time.time()
    spark = crear_sesion()
    spark.sparkContext.setLogLevel("ERROR")
    t_arranque = time.time() - t_inicio
    print(f"[02] Spark {spark.version} arrancado en {t_arranque:.1f} s")
    registrar_metrica("arranque_jvm", "2-spark-lectura", "spark", 0, t_arranque,
                      medir_memoria(), "costo fijo de levantar la JVM")

    # --- 2.1 Lectura del Parquet generado por Dask -------------------------
    t0 = time.time()
    df = spark.read.parquet(str(DIR_PARQUET))
    n = df.count()
    t_lectura = time.time() - t0
    print(f"[02] Leidas {n:,} filas desde el Parquet de Dask en {t_lectura:.1f} s")
    print(f"[02] Particiones de Spark: {df.rdd.getNumPartitions()}")
    registrar_metrica("lectura_parquet", "2-spark-lectura", "spark", n, t_lectura,
                      medir_memoria(), "entrada producida por Dask")

    # --- 2.2 Transformaciones y filtros distribuidos -----------------------
    t0 = time.time()
    limpio = (df
              .filter(F.col("bandas_validas") == 5)
              .filter(F.col("color_gr").isNotNull() & F.col("color_ri").isNotNull())
              .withColumn("magnitud_media",
                          (F.col("u") + F.col("g") + F.col("r") +
                           F.col("i") + F.col("z")) / 5)
              .withColumn("tipo_distancia",
                          F.when(F.col("redshift") < 0.1, "cercano")
                           .when(F.col("redshift") < 0.5, "intermedio")
                           .otherwise("lejano")))
    n_limpio = limpio.count()
    t_filtros = time.time() - t0
    print(f"[02] Tras filtros: {n_limpio:,} filas ({n_limpio/n:.1%}) en {t_filtros:.1f} s")
    registrar_metrica("filtros_transformaciones", "2-spark-transformacion", "spark",
                      n_limpio, t_filtros, medir_memoria())

    # --- 2.3 Agregacion distribuida ----------------------------------------
    t0 = time.time()
    resumen_clase = (limpio.groupBy("class")
                     .agg(F.count("*").alias("objetos"),
                          F.round(F.avg("redshift"), 4).alias("redshift_medio"),
                          F.round(F.stddev("redshift"), 4).alias("redshift_desv"),
                          F.round(F.avg("color_gr"), 4).alias("color_gr_medio"),
                          F.round(F.avg("color_ri"), 4).alias("color_ri_medio"),
                          F.round(F.avg("magnitud_media"), 3).alias("magnitud_media"))
                     .orderBy(F.desc("objetos")))
    resumen_clase.cache().count()
    t_agg = time.time() - t0
    print(f"[02] Agregacion por clase espectral ({t_agg:.1f} s):")
    resumen_clase.show(truncate=False)
    registrar_metrica("agregacion_por_clase", "2-spark-agregacion", "spark",
                      n_limpio, t_agg, medir_memoria(), "groupBy + 6 agregados")
    resumen_clase.toPandas().to_csv(DIR_METRICAS / "resumen_por_clase.csv", index=False)

    # --- 2.4 Analisis avanzado A: funciones de ventana ---------------------
    #
    # Para cada region del cielo, ordenar los objetos por brillo y quedarse con
    # los mas brillantes. Una ventana particionada por region es el tipo de
    # operacion que en pandas obligaria a un bucle por grupo.
    t0 = time.time()
    ventana = Window.partitionBy("region").orderBy(F.asc("magnitud_media"))
    ranking = (limpio
               .withColumn("rango_brillo", F.row_number().over(ventana))
               .withColumn("percentil_brillo", F.round(F.percent_rank().over(ventana), 4))
               .filter(F.col("rango_brillo") <= 10)
               .select("region", "rango_brillo", "obj_ID", "class",
                       "magnitud_media", "redshift"))
    ranking.cache().count()
    t_ventana = time.time() - t0
    print(f"[02] Funcion de ventana (top 10 por region) en {t_ventana:.1f} s")
    ranking.orderBy("region", "rango_brillo").show(8, truncate=False)
    registrar_metrica("funcion_ventana_top10", "2-spark-avanzado", "spark",
                      n_limpio, t_ventana, medir_memoria(),
                      "row_number + percent_rank particionado por region")
    ranking.orderBy("region", "rango_brillo").limit(50).toPandas().to_csv(
        DIR_METRICAS / "top_brillantes.csv", index=False)

    # --- 2.5 Analisis avanzado B: join entre dos conjuntos -----------------
    #
    # Se cruza el catalogo con la tabla de campos observacionales que produjo
    # Dask. La tabla de campos es pequena, asi que Spark puede difundirla a
    # todos los executors (broadcast join) en lugar de barajar la tabla grande.
    t0 = time.time()
    campos = spark.read.parquet(str(DIR_CAMPOS))
    n_campos = campos.count()
    print(f"[02] Tabla secundaria: {n_campos:,} campos observacionales")

    cruzado = limpio.join(F.broadcast(campos), on="field_ID", how="inner")
    calidad = (cruzado.groupBy("calidad")
               .agg(F.count("*").alias("objetos"),
                    F.round(F.avg("seeing_medio"), 3).alias("seeing_medio"),
                    F.round(F.avg("magnitud_media"), 3).alias("magnitud_media"),
                    F.round(F.stddev("color_gr"), 4).alias("dispersion_color"))
               .orderBy("seeing_medio"))
    calidad.cache().count()
    t_join = time.time() - t0
    print(f"[02] Join + agregacion en {t_join:.1f} s")
    calidad.show(truncate=False)
    registrar_metrica("join_broadcast_campos", "2-spark-avanzado", "spark",
                      n_limpio, t_join, medir_memoria(),
                      f"broadcast join catalogo x {n_campos} campos")
    calidad.toPandas().to_csv(DIR_METRICAS / "calidad_vs_fotometria.csv", index=False)

    # --- 2.6 Analisis avanzado C: Spark SQL --------------------------------
    t0 = time.time()
    cruzado.createOrReplaceTempView("catalogo")
    consulta = spark.sql("""
        SELECT  class                   AS clase,
                tipo_distancia          AS distancia,
                calidad                 AS calidad_campo,
                COUNT(*)                AS objetos,
                ROUND(AVG(redshift), 4) AS redshift_medio,
                ROUND(AVG(color_gr), 4) AS color_gr,
                ROUND(AVG(dist_mpc), 1) AS dist_mpc_media
        FROM    catalogo
        WHERE   redshift > 0
        GROUP BY class, tipo_distancia, calidad
        HAVING  COUNT(*) > 1000
        ORDER BY objetos DESC
    """)
    filas_sql = consulta.collect()
    t_sql = time.time() - t0
    print(f"[02] Spark SQL ({len(filas_sql)} grupos) en {t_sql:.1f} s")
    consulta.show(10, truncate=False)
    registrar_metrica("spark_sql_agrupado", "2-spark-avanzado", "spark",
                      n_limpio, t_sql, medir_memoria(), "GROUP BY 3 claves + HAVING")
    consulta.toPandas().to_csv(DIR_METRICAS / "consulta_sql.csv", index=False)

    # --- 2.7 Analisis avanzado D: modelo con MLlib -------------------------
    #
    # Clasificar cada objeto como estrella, galaxia o cuasar a partir de sus
    # colores. Es el problema central del articulo de referencia y la razon por
    # la que los indices de color se calcularon en la etapa de Dask.
    t0 = time.time()
    predictoras = ["color_ug", "color_gr", "color_ri", "color_iz",
                   "magnitud_media", "redshift"]

    # PROBLEMA METODOLOGICO QUE HAY QUE RESOLVER ANTES DE ENTRENAR
    # ------------------------------------------------------------
    # Cada objeto del cielo fue observado en las N epocas, asi que aparece N
    # veces en la tabla. Si partieramos las filas al azar, el mismo objeto
    # caeria en entrenamiento y en prueba a la vez: el modelo reconoceria un
    # objeto que ya vio y la exactitud saldria inflada. Es fuga de informacion.
    #
    # La solucion es la que usan los sondeos reales: construir un catalogo
    # coadicionado, promediando las epocas de cada objeto para obtener una
    # medicion mas profunda y con menos ruido. Eso deja una fila por objeto y
    # elimina la fuga de raiz. Ademas es otra agregacion distribuida, que es
    # justo donde Spark rinde.
    coadicion = (limpio.groupBy("obj_ID", "class")
                 .agg(F.avg("color_ug").alias("color_ug"),
                      F.avg("color_gr").alias("color_gr"),
                      F.avg("color_ri").alias("color_ri"),
                      F.avg("color_iz").alias("color_iz"),
                      F.avg("magnitud_media").alias("magnitud_media"),
                      F.avg("redshift").alias("redshift"),
                      F.count("*").alias("n_epocas")))
    coadicion.cache()
    n_objetos = coadicion.count()
    print(f"[02] Catalogo coadicionado: {n_objetos:,} objetos unicos "
          f"(desde {n_limpio:,} observaciones)")

    datos_ml = coadicion.select(*predictoras, "class")
    train, test = datos_ml.randomSplit([0.8, 0.2], seed=SEMILLA)

    etapas = [
        Imputer(inputCols=predictoras, outputCols=predictoras, strategy="median"),
        StringIndexer(inputCol="class", outputCol="etiqueta", handleInvalid="skip"),
        VectorAssembler(inputCols=predictoras, outputCol="caracteristicas",
                        handleInvalid="skip"),
        RandomForestClassifier(featuresCol="caracteristicas", labelCol="etiqueta",
                               numTrees=40, maxDepth=8, seed=SEMILLA),
    ]
    modelo = Pipeline(stages=etapas).fit(train)
    predicciones = modelo.transform(test)

    ev = MulticlassClassificationEvaluator(labelCol="etiqueta", predictionCol="prediction")
    exactitud = ev.setMetricName("accuracy").evaluate(predicciones)
    f1 = ev.setMetricName("f1").evaluate(predicciones)
    t_ml = time.time() - t0

    print(f"[02] MLlib RandomForest ({t_ml:.1f} s)")
    print(f"[02]   Exactitud : {exactitud:.4f}")
    print(f"[02]   F1        : {f1:.4f}")

    rf = modelo.stages[-1]
    importancias = sorted(zip(predictoras, rf.featureImportances.toArray()),
                          key=lambda x: -x[1])
    print("[02]   Importancia de variables:")
    for var, imp in importancias:
        print(f"[02]     {var:<16} {imp:.4f}")

    registrar_metrica("mllib_random_forest", "2-spark-avanzado", "spark",
                      n_objetos, t_ml, medir_memoria(),
                      f"coadicion {n_objetos} objetos, exactitud={exactitud:.4f}")

    pd.DataFrame(importancias, columns=["variable", "importancia"]).to_csv(
        DIR_METRICAS / "importancia_mllib.csv", index=False)
    pd.DataFrame([{"exactitud": exactitud, "f1": f1,
                   "n_objetos_coadicion": n_objetos,
                   "n_observaciones": n_limpio}]).to_csv(
        DIR_METRICAS / "metricas_mllib.csv", index=False)

    matriz = (predicciones.groupBy("class").pivot("prediction").count()
              .fillna(0).orderBy("class"))
    matriz.show(truncate=False)
    matriz.toPandas().to_csv(DIR_METRICAS / "matriz_confusion_mllib.csv", index=False)

    # --- 2.8 Salidas agregadas para las visualizaciones --------------------
    #
    # Se guardan resultados YA AGREGADOS. En ningun momento se convierte el
    # conjunto completo a pandas para graficar, que es justo lo que prohibe
    # la guia.
    t0 = time.time()

    (limpio.filter((F.col("redshift") > 0) & (F.col("redshift") < 3))
     .withColumn("bin", F.floor(F.col("redshift") * 40) / 40)
     .groupBy("bin", "class").count().orderBy("bin")
     .toPandas().to_csv(DIR_METRICAS / "hist_redshift.csv", index=False))

    (limpio.filter(F.col("color_gr").between(-1, 3) & F.col("color_ri").between(-1, 3))
     .withColumn("gr_bin", F.round(F.col("color_gr") * 20) / 20)
     .withColumn("ri_bin", F.round(F.col("color_ri") * 20) / 20)
     .groupBy("gr_bin", "ri_bin", "class").count()
     .filter(F.col("count") > 20)
     .toPandas().to_csv(DIR_METRICAS / "diagrama_color.csv", index=False))

    (limpio.groupBy("region")
     .agg(F.count("*").alias("objetos"),
          F.round(F.avg("redshift"), 4).alias("redshift_medio"),
          F.round(F.avg("magnitud_media"), 3).alias("magnitud_media"))
     .orderBy(F.desc("objetos"))
     .toPandas().to_csv(DIR_METRICAS / "por_region.csv", index=False))

    (coadicion.groupBy("class")
     .agg(F.count("*").alias("objetos"),
          F.round(F.avg("n_epocas"), 1).alias("epocas_medias"),
          F.round(F.avg("redshift"), 4).alias("redshift_medio"))
     .toPandas().to_csv(DIR_METRICAS / "coadicion_por_clase.csv", index=False))

    print(f"[02] Agregados para visualizacion exportados en {time.time()-t0:.1f} s")

    total = time.time() - t_inicio
    print(f"[02] Etapa Spark completa en {total:.1f} s")
    registrar_metrica("etapa_spark_completa", "2-spark-total", "spark", n_limpio,
                      total, medir_memoria(), "incluye arranque de la JVM")
    spark.stop()


if __name__ == "__main__":
    main()
