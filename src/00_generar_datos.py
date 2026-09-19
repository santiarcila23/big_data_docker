"""
Etapa 0 - Construccion del conjunto ampliado y particionado
===========================================================

La guia exige un conjunto "razonablemente grande" y admite explicitamente
"construir una version ampliada o particionada". Eso es lo que hace este script.

El punto de partida son datos REALES: el catalogo Stellar Classification SDSS17
(100.000 objetos del Sloan Digital Sky Survey Data Release 17, con fotometria en
cinco bandas, redshift espectroscopico y clase espectral).

La ampliacion no es una copia ciega. Simula lo que hace un sondeo real: observar
la misma region del cielo en varias noches. Cada "epoca" vuelve a medir los
mismos objetos con ruido fotometrico propio de la noche, de modo que:

  - las magnitudes varian ligeramente entre epocas, como en la realidad;
  - las posiciones y la clase espectral se mantienen, porque son propiedades
    del objeto y no de la observacion;
  - cada campo del cielo tiene ademas una calidad intrinseca (los campos
    observados a baja altura atraviesan mas atmosfera), lo que da variedad
    real a la tabla secundaria que Spark usara en el join;
  - el resultado es un conjunto multi-epoca que justifica de verdad el
    particionamiento y las funciones de ventana en Spark.

Se genera un archivo CSV por epoca, para que la ingesta con Dask lea multiples
archivos, tal como pide la guia.
"""
import time
import numpy as np
import pandas as pd

from config import (CSV_SDSS, DIR_EPOCAS, N_EPOCAS, ESCALA, SEMILLA,
                    BANDAS, CENTINELA, registrar_metrica)


def main():
    t0 = time.time()
    print(f"[00] Escala '{ESCALA}' -> {N_EPOCAS} epocas de observacion")

    base = pd.read_csv(CSV_SDSS)
    print(f"[00] Catalogo real cargado: {base.shape[0]:,} objetos, {base.shape[1]} columnas")

    # Nos quedamos con las columnas con sentido fisico o de identificacion.
    # run_ID, rerun_ID, cam_col, plate, MJD y fiber_ID son metadatos del
    # instrumento; conservamos field_ID porque lo usaremos para el join.
    columnas = ["obj_ID", "alpha", "delta"] + BANDAS + ["class", "redshift", "field_ID"]
    base = base[columnas].copy()

    DIR_EPOCAS.mkdir(parents=True, exist_ok=True)
    for viejo in DIR_EPOCAS.glob("*.csv"):
        viejo.unlink()

    rng = np.random.default_rng(SEMILLA)

    # Calidad intrinseca de cada campo del cielo. Es propiedad del campo, no de
    # la noche, y es lo que hara que la tabla de campos tenga variedad real.
    campos_unicos = base["field_ID"].unique()
    factor_campo = dict(zip(campos_unicos,
                            rng.lognormal(0.0, 0.28, size=len(campos_unicos))))
    base["_factor"] = base["field_ID"].map(factor_campo)

    total = 0
    for epoca in range(1, N_EPOCAS + 1):
        df = base.copy()

        # Calidad de la noche: unas noches son mejores que otras.
        seeing_noche = rng.uniform(0.75, 1.6)
        seeing_fila = np.clip(seeing_noche * df["_factor"].to_numpy(), 0.5, 3.5)

        for banda in BANDAS:
            valores = df[banda].to_numpy(dtype=float)
            validos = valores != CENTINELA
            # El ruido fotometrico crece con la magnitud (los objetos debiles se
            # miden peor) y con el seeing efectivo de esa noche en ese campo.
            sigma = 0.01 + 0.004 * seeing_fila * np.clip(valores - 15.0, 0, None)
            ruido = rng.normal(0.0, 1.0, size=len(valores)) * sigma
            valores = np.where(validos, valores + ruido, CENTINELA)

            # Algunas mediciones se pierden cada noche (nubes, pixeles saturados,
            # objeto fuera del campo). Se marcan con el centinela.
            perdidas = rng.random(len(valores)) < (0.004 * seeing_fila)
            valores = np.where(perdidas, CENTINELA, valores)
            df[banda] = valores

        df["epoca"] = epoca
        df["seeing"] = np.round(seeing_fila, 3)
        df["mjd_obs"] = 58000 + epoca * 3      # fecha juliana modificada ficticia
        df = df.drop(columns=["_factor"])

        df.to_csv(DIR_EPOCAS / f"epoca_{epoca:03d}.csv", index=False)
        total += len(df)
        if epoca % 10 == 0 or epoca == N_EPOCAS:
            print(f"[00]   epoca {epoca:>3}/{N_EPOCAS}  ({total:,} filas acumuladas)")

    tam_mb = sum(f.stat().st_size for f in DIR_EPOCAS.glob("*.csv")) / 1e6
    dt = time.time() - t0
    print(f"[00] Listo: {N_EPOCAS} archivos, {total:,} filas, {tam_mb:,.0f} MB en {dt:.1f} s")
    registrar_metrica("generacion_datos", "0-generacion", "pandas", total, dt,
                      nota=f"{N_EPOCAS} archivos CSV, {tam_mb:.0f} MB")


if __name__ == "__main__":
    main()
