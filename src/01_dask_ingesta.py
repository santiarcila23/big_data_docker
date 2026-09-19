"""
Etapa 1 - Ingesta, limpieza y transformacion con Dask
=====================================================

Por que Dask en esta etapa y no Spark
-------------------------------------
Esta etapa es de entrada/salida y de limpieza fila a fila: leer decenas de CSV,
convertir centinelas a nulos, calcular columnas nuevas y escribir Parquet. Son
operaciones que se aplican de forma independiente a cada particion y no
requieren mover datos entre nodos (no hay shuffle).

Para ese perfil Dask es la eleccion adecuada por tres razones concretas:

1. Costo de arranque. Dask levanta en milisegundos dentro del mismo proceso de
   Python. Spark tiene que iniciar una JVM, un driver y los executors, lo que
   cuesta varios segundos aunque el trabajo sea trivial.
2. Integracion con el ecosistema cientifico. El codigo de limpieza es
   basicamente pandas y NumPy. En Dask se reutiliza tal cual; en Spark habria
   que reescribirlo con la API de DataFrames o pagar el sobrecosto de las UDF
   de Python, que serializan datos entre la JVM y el interprete.
3. Control fino del particionamiento de salida. Escribimos Parquet particionado
   por region del cielo, que es el mismo esquema que usa HATS/LSDB en el
   articulo de referencia (Caplar et al. 2025).

Lo que NO hace bien Dask, y por eso la siguiente etapa es Spark: agregaciones
con mucho shuffle, joins grandes y SQL distribuido.
"""
import time
import shutil
import numpy as np
import dask.dataframe as dd

from config import (DIR_EPOCAS, DIR_PARQUET, DIR_CAMPOS, BANDAS, CENTINELA,
                    ESCALA, registrar_metrica, medir_memoria)

# Tipos explicitos. Sin esto Dask infiere leyendo solo el principio del archivo
# y falla mas adelante si una columna cambia de aspecto.
TIPOS = {
    "obj_ID": "float64", "alpha": "float64", "delta": "float64",
    "u": "float64", "g": "float64", "r": "float64", "i": "float64", "z": "float64",
    "class": "object", "redshift": "float64", "field_ID": "int64",
    "epoca": "int64", "seeing": "float64", "mjd_obs": "int64",
}


def region_cielo(alpha, delta):
    """
    Divide el cielo en celdas de 30 x 30 grados y devuelve un identificador.

    Es una version simplificada del particionamiento espacial jerarquico que usa
    HATS: agrupar objetos vecinos en el cielo dentro del mismo archivo, para que
    las consultas por zona lean pocos archivos en lugar de todos.
    """
    banda_alpha = (alpha // 30).astype("int64")
    banda_delta = ((delta + 90) // 30).astype("int64")
    return banda_alpha * 10 + banda_delta


def main():
    print(f"[01] DASK - ingesta y transformacion (escala '{ESCALA}')")
    mem0 = medir_memoria()

    # --- 1.1 Ingesta particionada -----------------------------------------
    t0 = time.time()
    ddf = dd.read_csv(str(DIR_EPOCAS / "epoca_*.csv"), dtype=TIPOS, blocksize="32MB")
    t_perezosa = time.time() - t0
    print(f"[01] Particiones: {ddf.npartitions}")
    print(f"[01] read_csv devolvio en {t_perezosa:.3f} s "
          f"(evaluacion perezosa: todavia no leyo nada)")

    t0 = time.time()
    n_filas = len(ddf)
    t_conteo = time.time() - t0
    print(f"[01] Filas: {n_filas:,}  (el conteo tardo {t_conteo:.1f} s: aqui si leyo)")
    registrar_metrica("ingesta_conteo", "1-dask-ingesta", "dask", n_filas, t_conteo,
                      medir_memoria(), f"{ddf.npartitions} particiones")

    # --- 1.2 Limpieza ------------------------------------------------------
    t0 = time.time()

    # El catalogo marca las mediciones no disponibles con -9999. Si se deja ese
    # valor, cualquier promedio queda destruido. Se convierte a nulo.
    centinelas = sum((ddf[b] <= CENTINELA + 1).sum() for b in BANDAS).compute()
    print(f"[01] Centinelas -9999 convertidos a nulo: {centinelas:,}")
    for banda in BANDAS:
        ddf[banda] = ddf[banda].mask(ddf[banda] <= CENTINELA + 1, np.nan)

    # Rangos fisicamente plausibles. Fuera de ellos es error de medicion.
    ddf = ddf[(ddf["redshift"] > -0.01) & (ddf["redshift"] < 8.0)]
    for banda in BANDAS:
        ddf[banda] = ddf[banda].mask((ddf[banda] < 5) | (ddf[banda] > 35), np.nan)

    # Una fila sin ninguna banda valida no aporta nada: se descarta.
    ddf = ddf.dropna(subset=BANDAS, how="all")

    # --- 1.3 Variables derivadas (la guia pide al menos dos) --------------
    #
    # (a) Indices de color. Son la diferencia de magnitud entre dos bandas y
    #     constituyen la variable fundamental de la astronomia fotometrica:
    #     miden la pendiente del espectro y separan estrellas, galaxias y
    #     cuasares mucho mejor que las magnitudes por separado.
    ddf["color_ug"] = ddf["u"] - ddf["g"]
    ddf["color_gr"] = ddf["g"] - ddf["r"]
    ddf["color_ri"] = ddf["r"] - ddf["i"]
    ddf["color_iz"] = ddf["i"] - ddf["z"]

    # (b) Region del cielo, para el particionamiento espacial de la salida.
    ddf["region"] = region_cielo(ddf["alpha"], ddf["delta"])

    # (c) Distancia comovil aproximada, en megaparsecs.
    #     Aproximacion de Hubble: d = c*z / H0. Es una simplificacion
    #     deliberada (no vale para redshift alto) y queda documentada.
    C_LUZ, H0 = 299792.458, 70.0
    ddf["dist_mpc"] = (C_LUZ * ddf["redshift"]) / H0

    # (d) Calidad de la observacion: cuantas bandas se midieron bien.
    ddf["bandas_validas"] = sum(ddf[b].notnull().astype("int8") for b in BANDAS)

    columnas = (["obj_ID", "alpha", "delta", "region"] + BANDAS +
                ["color_ug", "color_gr", "color_ri", "color_iz",
                 "class", "redshift", "dist_mpc", "field_ID",
                 "epoca", "seeing", "mjd_obs", "bandas_validas"])
    ddf = ddf[columnas]

    # --- 1.4 Salida en Parquet particionado --------------------------------
    #
    # Parquet y no CSV porque:
    #   - es columnar: Spark puede leer solo las columnas que necesita;
    #   - esta comprimido (snappy), lo que reduce la lectura de disco;
    #   - guarda el esquema, asi que Spark no tiene que inferir tipos;
    #   - es el formato que usan HATS/LSDB en el articulo de referencia.
    if DIR_PARQUET.exists():
        shutil.rmtree(DIR_PARQUET)

    ddf.to_parquet(str(DIR_PARQUET), engine="pyarrow", compression="snappy",
                   partition_on=["region"], write_index=False)

    # SEGUNDA SALIDA: version compacta, sin particionar por region.
    # ------------------------------------------------------------------
    # Particionar por region crea un archivo por cada combinacion de region y
    # particion de entrada: 45 regiones x 30 particiones = 1.350 archivos de
    # unos 100 KB cada uno. Eso es el "problema de los archivos pequenos": cada
    # archivo tiene un costo fijo de apertura y de lectura de metadatos, asi que
    # un escaneo completo se vuelve mas lento.
    #
    # El particionamiento sigue siendo correcto para consultas FILTRADAS por
    # region (Spark salta los archivos que no necesita). Pero para escaneos
    # completos conviene una version compacta. Escribimos las dos y en el
    # informe se compara cual conviene en cada caso.
    from config import DIR_INTERIM
    dir_compacto = DIR_INTERIM / "catalogo_parquet_compacto"
    if dir_compacto.exists():
        shutil.rmtree(dir_compacto)
    ddf.repartition(npartitions=8).to_parquet(
        str(dir_compacto), engine="pyarrow", compression="snappy", write_index=False)

    t_transf = time.time() - t0
    n_final = len(dd.read_parquet(str(DIR_PARQUET)))
    n_part = len(list(DIR_PARQUET.rglob("*.parquet")))
    n_comp = len(list(dir_compacto.rglob("*.parquet")))
    print(f"[01] Parquet particionado: {n_part} archivos | compacto: {n_comp} archivos")
    tam_csv = sum(f.stat().st_size for f in DIR_EPOCAS.glob("*.csv")) / 1e6
    tam_pq = sum(f.stat().st_size for f in DIR_PARQUET.rglob("*.parquet")) / 1e6

    print(f"[01] Limpieza + transformacion + escritura: {t_transf:.1f} s")
    print(f"[01] Filas conservadas: {n_final:,} de {n_filas:,} ({n_final/n_filas:.1%})")
    print(f"[01] Mediciones anuladas: {centinelas:,} "
          f"({centinelas/(n_filas*len(BANDAS)):.2%} del total)")
    print(f"[01] CSV entrada  : {tam_csv:,.0f} MB")
    print(f"[01] Parquet salida: {tam_pq:,.0f} MB (reduccion del {(1-tam_pq/tam_csv)*100:.0f}%)")
    registrar_metrica("limpieza_transformacion_parquet", "1-dask-transformacion",
                      "dask", n_final, t_transf, medir_memoria(),
                      f"CSV {tam_csv:.0f}MB -> Parquet {tam_pq:.0f}MB")

    # --- 1.5 Tabla secundaria, para el join distribuido en Spark -----------
    #
    # Metadatos por campo observacional. En un sondeo real esta tabla viene de
    # otra fuente (el registro del telescopio); aqui la derivamos del propio
    # catalogo. Es la segunda tabla que Spark cruzara con la principal.
    t0 = time.time()
    campos = (dd.read_parquet(str(DIR_PARQUET),
                              columns=["field_ID", "seeing", "bandas_validas"])
              .groupby("field_ID")
              .agg({"seeing": "mean", "bandas_validas": "mean"})
              .reset_index()
              .rename(columns={"seeing": "seeing_medio",
                               "bandas_validas": "completitud_media"}))
    campos_pd = campos.compute()
    campos_pd["calidad"] = np.where(campos_pd["seeing_medio"] < 1.3, "buena",
                             np.where(campos_pd["seeing_medio"] < 1.8, "media", "pobre"))
    if DIR_CAMPOS.exists():
        shutil.rmtree(DIR_CAMPOS)
    DIR_CAMPOS.mkdir(parents=True, exist_ok=True)
    campos_pd.to_parquet(DIR_CAMPOS / "campos.parquet", index=False)

    dt = time.time() - t0
    print(f"[01] Tabla de campos: {len(campos_pd):,} campos en {dt:.1f} s")
    print(campos_pd["calidad"].value_counts().to_string())
    registrar_metrica("tabla_campos", "1-dask-transformacion", "dask",
                      len(campos_pd), dt, medir_memoria(), "tabla secundaria para join")

    print(f"[01] Memoria del proceso: {mem0:.0f} MB -> {medir_memoria():.0f} MB")


if __name__ == "__main__":
    main()
