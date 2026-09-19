"""
Etapa 5 - Comparacion Dask vs Spark sobre la MISMA operacion
============================================================

La guia pide resolver al menos una operacion equivalente en los dos motores,
medir tiempos y discutir las diferencias. Aqui se comparan tres operaciones de
naturaleza distinta, porque el resultado no es el mismo en todas:

  A) Agregacion simple    groupBy(clase) + promedios        poco shuffle
  B) Agregacion compuesta groupBy(2 claves)                 shuffle medio
  C) Join + agregacion    catalogo x campos observacionales shuffle fuerte

Cada una se mide con dos tamanos de datos, para poder hablar de escalabilidad y
no solo de un punto suelto.

CONDICIONES DE JUSTICIA DE LA MEDICION
--------------------------------------
Una comparacion mal montada da numeros que no significan nada. Estas son las
precauciones que se tomaron, y conviene poder explicarlas en la sustentacion:

1. Los dos motores leen EL MISMO Parquet y EXACTAMENTE LAS MISMAS COLUMNAS.
   En una primera version dejamos que Dask leyera las 20 columnas mientras
   Spark, gracias a su optimizador, solo leia las necesarias. Eso inflaba la
   ventaja de Spark de forma artificial.

2. Los datos quedan materializados en memoria en ambos lados antes de medir:
   .cache() en Spark y .persist() en Dask. Sin esto, Spark reutilizaba su cache
   entre repeticiones mientras Dask releia el disco cada vez.

3. El arranque de la JVM de Spark se mide aparte y NO se suma a las operaciones,
   porque es un costo fijo por sesion. Se reporta por separado, que es lo
   honesto: en trabajos cortos ese costo puede pesar mas que todo el calculo.

4. Se descarta una ejecucion de calentamiento y se reporta el mejor tiempo de
   las siguientes, para reducir el ruido del sistema operativo.
"""
import time
import pandas as pd
import dask.dataframe as dd

from pyspark.sql import SparkSession, functions as F

from config import (DIR_PARQUET, DIR_CAMPOS, DIR_METRICAS, ESCALA,
                    SPARK_MEMORIA_DRIVER, SPARK_PARTICIONES,
                    registrar_metrica, medir_memoria)

# Solo las columnas que las tres operaciones necesitan. Igual para ambos motores.
COLUMNAS = ["class", "redshift", "color_gr", "obj_ID", "region",
            "field_ID", "bandas_validas"]

RESULTADOS = []


def cronometrar(fn, repeticiones=2):
    """Una ejecucion de calentamiento y luego el mejor de N tiempos."""
    fn()                                   # calentamiento, no se cuenta
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.time()
        fn()
        tiempos.append(time.time() - t0)
    return min(tiempos)


def registrar_par(operacion, etiqueta, filas, ta, tb):
    ganador = "Dask" if ta < tb else "Spark"
    factor = max(ta, tb) / min(ta, tb)
    print(f"[05] {operacion[:36]:<36} Dask {ta:7.2f} s | Spark {tb:7.2f} s"
          f"  -> {ganador} {factor:.2f}x")
    RESULTADOS.append({"operacion": operacion, "tamano": etiqueta, "filas": filas,
                       "dask_seg": round(ta, 2), "spark_seg": round(tb, 2),
                       "ganador": ganador, "factor": round(factor, 2)})
    clave = "comp_" + operacion.split(".")[0]
    registrar_metrica(clave, "5-comparacion", "dask", filas, ta,
                      medir_memoria(), f"tamano {etiqueta}")
    registrar_metrica(clave, "5-comparacion", "spark", filas, tb,
                      medir_memoria(), f"tamano {etiqueta}")


def main():
    print(f"[05] COMPARACION DASK vs SPARK (escala base '{ESCALA}')")
    print(f"[05] Columnas leidas por ambos motores: {len(COLUMNAS)}")

    t0 = time.time()
    spark = (SparkSession.builder
             .appName("ComparacionDaskSpark")
             .master("local[*]")
             .config("spark.driver.memory", SPARK_MEMORIA_DRIVER)
             .config("spark.sql.shuffle.partitions", str(SPARK_PARTICIONES))
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    t_arranque = time.time() - t0
    print(f"[05] Arranque de la JVM de Spark: {t_arranque:.2f} s (costo fijo por sesion)")

    t0 = time.time()
    _ = dd.read_parquet(str(DIR_PARQUET), columns=["class"])
    t_dask_arr = time.time() - t0
    print(f"[05] Equivalente en Dask: {t_dask_arr:.3f} s (no hay JVM que levantar)")
    registrar_metrica("arranque_motor", "5-comparacion", "spark", 0, t_arranque,
                      medir_memoria(), "costo fijo de la JVM")
    registrar_metrica("arranque_motor", "5-comparacion", "dask", 0, t_dask_arr,
                      medir_memoria(), "sin JVM")

    campos_pd = pd.read_parquet(str(DIR_CAMPOS))[["field_ID", "calidad"]]
    campos_sp = spark.read.parquet(str(DIR_CAMPOS)).select("field_ID", "calidad")

    for etiqueta, fraccion in [("50%", 0.5), ("100%", 1.0)]:
        print(f"\n[05] {'='*62}")
        print(f"[05] TAMANO: {etiqueta} del conjunto")
        print(f"[05] {'='*62}")

        t0 = time.time()
        ddf = dd.read_parquet(str(DIR_PARQUET), columns=COLUMNAS)
        if fraccion < 1.0:
            ddf = ddf.sample(frac=fraccion, random_state=42)
        ddf = ddf[ddf["bandas_validas"] == 5].persist()
        n_dask = len(ddf)
        t_prep_d = time.time() - t0

        t0 = time.time()
        sdf = spark.read.parquet(str(DIR_PARQUET)).select(*COLUMNAS)
        if fraccion < 1.0:
            sdf = sdf.sample(fraction=fraccion, seed=42)
        sdf = sdf.filter(F.col("bandas_validas") == 5).cache()
        n_spark = sdf.count()
        t_prep_s = time.time() - t0

        print(f"[05] Filas - Dask: {n_dask:,} (materializar {t_prep_d:.1f} s) | "
              f"Spark: {n_spark:,} (materializar {t_prep_s:.1f} s)")

        def dask_a():
            return (ddf.groupby("class")
                    .agg({"redshift": "mean", "color_gr": "mean", "obj_ID": "count"})
                    .compute())

        def spark_a():
            return (sdf.groupBy("class")
                    .agg(F.avg("redshift"), F.avg("color_gr"), F.count("obj_ID"))
                    .collect())

        registrar_par("A. Agregacion simple (groupBy 1 clave)", etiqueta, n_spark,
                      cronometrar(dask_a), cronometrar(spark_a))

        def dask_b():
            d = ddf[ddf["redshift"] > 0]
            return (d.groupby(["class", "region"])
                    .agg({"redshift": "mean", "obj_ID": "count"})
                    .compute())

        def spark_b():
            return (sdf.filter(F.col("redshift") > 0)
                    .groupBy("class", "region")
                    .agg(F.avg("redshift"), F.count("obj_ID"))
                    .collect())

        registrar_par("B. Agregacion compuesta (2 claves)", etiqueta, n_spark,
                      cronometrar(dask_b), cronometrar(spark_b))

        def dask_c():
            unido = ddf.merge(campos_pd, on="field_ID", how="inner")
            return (unido.groupby("calidad")
                    .agg({"redshift": "mean", "obj_ID": "count"})
                    .compute())

        def spark_c():
            unido = sdf.join(F.broadcast(campos_sp), on="field_ID", how="inner")
            return (unido.groupBy("calidad")
                    .agg(F.avg("redshift"), F.count("obj_ID"))
                    .collect())

        registrar_par("C. Join + agregacion", etiqueta, n_spark,
                      cronometrar(dask_c, repeticiones=1),
                      cronometrar(spark_c, repeticiones=1))

        sdf.unpersist()
        del ddf

    tabla = pd.DataFrame(RESULTADOS)
    tabla.to_csv(DIR_METRICAS / "comparacion_dask_spark.csv", index=False)
    print("\n[05] TABLA COMPARATIVA")
    print(tabla.to_string(index=False))

    pd.DataFrame([{"motor": "Spark", "arranque_seg": round(t_arranque, 2)},
                  {"motor": "Dask", "arranque_seg": round(t_dask_arr, 3)}]).to_csv(
        DIR_METRICAS / "arranque_motores.csv", index=False)

    print(f"\n[05] El arranque de Spark ({t_arranque:.1f} s) no esta incluido arriba.")
    spark.stop()


if __name__ == "__main__":
    main()
