"""
Configuracion central del proyecto.

Todas las rutas y parametros se definen aqui para que ningun script traiga
valores incrustados. El tamano del conjunto se controla con la variable de
entorno ESCALA, lo que permite ejecutar el mismo codigo en dos configuraciones
distintas sin tocar el codigo (requisito de la guia: medir al menos dos
configuraciones o tamanos de datos).
"""
from pathlib import Path
import os

RAIZ = Path(__file__).resolve().parent.parent

DIR_RAW      = RAIZ / "data" / "raw"
DIR_INTERIM  = RAIZ / "data" / "interim"
DIR_OUTPUTS  = RAIZ / "outputs"
DIR_FIGURAS  = DIR_OUTPUTS / "figuras"
DIR_METRICAS = DIR_OUTPUTS / "metricas"

for d in (DIR_INTERIM, DIR_FIGURAS, DIR_METRICAS):
    d.mkdir(parents=True, exist_ok=True)

# --- Datos -----------------------------------------------------------------
CSV_SDSS    = DIR_RAW / "star_classification.csv"   # catalogo real SDSS DR17
DIR_EPOCAS  = DIR_INTERIM / "epocas"                # catalogo ampliado, CSV particionado
DIR_PARQUET = DIR_INTERIM / "catalogo_parquet"      # salida de Dask, entrada de Spark
DIR_CAMPOS  = DIR_INTERIM / "campos_parquet"        # tabla secundaria para el join
DIR_MEDICO  = DIR_INTERIM / "medico_parquet"        # caso medico

# --- Escala del experimento ------------------------------------------------
# Cada "epoca de observacion" son 100.000 objetos del catalogo real.
#   pequena =  4 epocas ~ 0,4 M filas   (prueba rapida, la que usa Docker)
#   media   = 12 epocas ~ 1,2 M filas
#   grande  = 30 epocas ~ 3,0 M filas   (medicion principal)
ESCALAS  = {"pequena": 4, "media": 12, "grande": 30}
ESCALA   = os.environ.get("ESCALA", "grande").lower()
N_EPOCAS = ESCALAS.get(ESCALA, 30)

SEMILLA = 42

# --- Spark -----------------------------------------------------------------
SPARK_MEMORIA_DRIVER = os.environ.get("SPARK_DRIVER_MEM", "2g")
SPARK_PARTICIONES    = int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))

# --- Dominio ---------------------------------------------------------------
# Las cinco bandas fotometricas del SDSS (del ultravioleta al infrarrojo cercano)
BANDAS = ["u", "g", "r", "i", "z"]

# El catalogo usa -9999 como centinela de "medicion no disponible".
# No es un valor fisico: hay que convertirlo a nulo antes de cualquier calculo.
CENTINELA = -9999.0


def registrar_metrica(nombre, etapa, motor, filas, segundos, memoria_mb=None, nota=""):
    """Agrega una fila al registro de tiempos que alimenta la tabla comparativa."""
    import csv
    from datetime import datetime
    archivo = DIR_METRICAS / "tiempos.csv"
    nuevo = not archivo.exists()
    with open(archivo, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["timestamp", "operacion", "etapa", "motor", "escala",
                        "filas", "segundos", "memoria_pico_mb", "nota"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), nombre, etapa,
                    motor, ESCALA, filas, round(segundos, 3),
                    "" if memoria_mb is None else round(memoria_mb, 1), nota])


def medir_memoria():
    """Memoria residente del proceso actual, en MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None
