"""
Orquestador del pipeline completo
=================================

Ejecuta todas las etapas en orden y deja un resumen al final. Es el comando de
inicio del contenedor Docker.

    python src/run_pipeline.py                 # todo
    python src/run_pipeline.py --solo 1 2      # solo las etapas 1 y 2
    ESCALA=grande python src/run_pipeline.py   # con el conjunto completo

El orden importa: la etapa 2 (Spark) no puede correr si la etapa 1 (Dask) no
produjo el Parquet. Esa dependencia es justamente el intercambio de datos entre
los dos motores que exige la guia.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from config import ESCALA, N_EPOCAS, DIR_METRICAS, DIR_FIGURAS  # noqa: E402

ETAPAS = [
    ("0", "00_generar_datos.py",        "Construccion del conjunto ampliado"),
    ("1", "01_dask_ingesta.py",         "Dask: ingesta, limpieza, Parquet"),
    ("2", "02_spark_procesamiento.py",  "Spark: agregaciones, ventana, join, SQL, MLlib"),
    ("3", "03_algoritmo_genetico.py",   "Algoritmo genetico: secuencial vs Dask"),
    ("4", "04_caso_medico.py",          "Caso medico con cohorte sintetica"),
    ("5", "05_comparacion.py",          "Comparacion Dask vs Spark"),
    ("6", "06_visualizaciones.py",      "Visualizaciones"),
]


def ejecutar(archivo):
    ruta = AQUI / archivo
    spec = importlib.util.spec_from_file_location(ruta.stem, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    modulo.main()


def main():
    ap = argparse.ArgumentParser(description="Pipeline integrador Dask + Spark")
    ap.add_argument("--solo", nargs="*", default=None,
                    help="numeros de etapa a ejecutar, por ejemplo: --solo 1 2")
    args = ap.parse_args()

    etapas = ETAPAS if not args.solo else [e for e in ETAPAS if e[0] in args.solo]

    print("=" * 74)
    print("  PRACTICA INTEGRADORA DE BIG DATA - Dask + Spark + Docker")
    print("  Institucion Universitaria de Envigado")
    print(f"  Escala: '{ESCALA}' ({N_EPOCAS} epocas ~ {N_EPOCAS*100_000:,} filas)")
    print("=" * 74)

    t_total = time.time()
    resumen = []

    for num, archivo, descripcion in etapas:
        print(f"\n{'-'*74}\n  ETAPA {num} - {descripcion}\n{'-'*74}")
        t0 = time.time()
        try:
            ejecutar(archivo)
            estado = "OK"
        except Exception as e:
            estado = f"ERROR: {type(e).__name__}: {e}"
            print(f"  !! {estado}")
        dt = time.time() - t0
        resumen.append((num, descripcion, estado, dt))
        print(f"  Etapa {num} terminada en {dt:.1f} s [{estado.split(':')[0]}]")

    print(f"\n{'='*74}\n  RESUMEN\n{'='*74}")
    for num, desc, estado, dt in resumen:
        marca = "OK " if estado == "OK" else "FALLO"
        print(f"  [{marca}] Etapa {num}: {desc:<48} {dt:7.1f} s")
    print(f"\n  Tiempo total: {time.time()-t_total:.1f} s")
    print(f"  Metricas  -> {DIR_METRICAS}")
    print(f"  Figuras   -> {DIR_FIGURAS}")
    print("=" * 74)


if __name__ == "__main__":
    main()
