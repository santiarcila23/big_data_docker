"""
Actividad 3.3 - Caso cientifico: datos medicos
==============================================

ARTICULO DE REFERENCIA
----------------------
Chen, S. et al. (2025). "Machine Learning-Based Prediction of ICU Mortality in
Sepsis-Associated Acute Kidney Injury Patients Using MIMIC-IV Database with
Validation from eICU Database". arXiv:2502.17978.

Base de datos original: Johnson, A. E. W. et al. (2023). "MIMIC-IV, a freely
accessible electronic health record dataset". Scientific Data 10, 1.
Validacion externa: Pollard, T. J. et al. (2018). "The eICU Collaborative
Research Database". Scientific Data 5.

ANALISIS DEL ARTICULO
---------------------
Problema. La lesion renal aguda asociada a sepsis (SA-AKI) tiene mortalidad alta
en cuidados intensivos. Identificar temprano a los pacientes de mayor riesgo
permitiria intervenir antes.

Fuente y volumen. MIMIC-IV reune los registros clinicos de todos los ingresos al
Beth Israel Deaconess Medical Center de Boston entre 2008 y 2022: del orden de
300.000 pacientes y 70.000 estancias en UCI, con signos vitales, laboratorios,
medicacion y desenlaces. El estudio identifico 9.474 pacientes con SA-AKI.

Tecnologias y metodologia. Seleccion de variables con factor de inflacion de la
varianza (VIF) y eliminacion recursiva de caracteristicas (RFE), reduciendo a 24
predictoras; modelo XGBoost con busqueda en rejilla de hiperparametros;
interpretabilidad con SHAP y LIME; validacion externa en eICU.

Resultados. El modelo alcanza buena discriminacion y la interpretabilidad
senala el lactato serico, la puntuacion APACHE II, la diuresis total y el calcio
serico como las variables mas influyentes.

POR QUE NO SE REPRODUCE EXACTAMENTE
-----------------------------------
MIMIC-IV no es de descarga libre: exige credencial de PhysioNet, formacion
certificada en investigacion con sujetos humanos y firma de un acuerdo de uso de
datos. La guia advierte ademas que debemos respetar la privacidad y usar datos
abiertos, anonimizados o sinteticos.

ADAPTACION
----------
Construimos una cohorte SINTETICA de pacientes de UCI que reproduce la
ESTRUCTURA del estudio, no sus datos. Ningun registro corresponde a una persona
real; se generan con un modelo probabilistico donde las relaciones entre
variables y desenlace se fijan segun lo reportado en la literatura clinica
(lactato alto, diuresis baja y APACHE II alto aumentan el riesgo). Eso permite
replicar el flujo metodologico completo (limpieza, seleccion de variables,
modelo, interpretabilidad) sin tocar informacion clinica identificable.

REQUISITO TECNICO: el preprocesamiento usa Dask para lectura particionada,
limpieza y calculo paralelo, y guarda los resultados intermedios en Parquet.
"""
import time
import shutil
import numpy as np
import pandas as pd
import dask.dataframe as dd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

from config import (DIR_INTERIM, DIR_MEDICO, DIR_METRICAS, SEMILLA,
                    registrar_metrica, medir_memoria)

N_PACIENTES = 400_000     # cohorte sintetica, repartida en varios archivos
N_ARCHIVOS = 8
DIR_MED_RAW = DIR_INTERIM / "medico_csv"


def generar_cohorte():
    """
    Genera la cohorte sintetica en varios archivos, para que Dask haga ingesta
    particionada. NINGUN dato corresponde a un paciente real.
    """
    print(f"[04] Generando cohorte sintetica: {N_PACIENTES:,} estancias en UCI")
    DIR_MED_RAW.mkdir(parents=True, exist_ok=True)
    for viejo in DIR_MED_RAW.glob("*.csv"):
        viejo.unlink()

    rng = np.random.default_rng(SEMILLA)
    por_archivo = N_PACIENTES // N_ARCHIVOS

    for k in range(N_ARCHIVOS):
        n = por_archivo
        edad = np.clip(rng.normal(64, 16, n), 18, 98)
        # Variables clinicas con las distribuciones tipicas de UCI
        lactato = np.clip(rng.lognormal(0.55, 0.62, n), 0.3, 25)      # mmol/L
        creatinina = np.clip(rng.lognormal(0.35, 0.65, n), 0.2, 15)   # mg/dL
        diuresis = np.clip(rng.lognormal(6.6, 0.75, n), 10, 6000)     # mL/24h
        apache = np.clip(rng.normal(17, 8, n), 0, 71)                 # puntos
        frec_card = np.clip(rng.normal(95, 22, n), 30, 200)
        pa_media = np.clip(rng.normal(74, 15, n), 30, 140)
        leucocitos = np.clip(rng.lognormal(2.5, 0.55, n), 0.1, 80)    # x10^9/L
        plaquetas = np.clip(rng.normal(205, 95, n), 5, 900)
        calcio = np.clip(rng.normal(8.4, 0.9, n), 5, 13)              # mg/dL
        sofa = np.clip(rng.normal(6, 3.4, n), 0, 24)

        # Riesgo segun las relaciones reportadas en la literatura clinica.
        # Los coeficientes NO salen del articulo: se eligen para que el ejercicio
        # tenga senal aprendible y quede documentado que son sinteticos.
        logito = (-4.1
                  + 0.20 * lactato
                  + 0.055 * apache
                  + 0.095 * sofa
                  + 0.22 * creatinina
                  - 0.00042 * diuresis
                  + 0.021 * (edad - 60) / 10
                  - 0.16 * (calcio - 8.4)
                  - 0.012 * (pa_media - 74)
                  + rng.normal(0, 0.55, n))
        prob = 1 / (1 + np.exp(-logito))
        fallecido = (rng.random(n) < prob).astype(int)

        df = pd.DataFrame({
            "id_estancia": np.arange(k * por_archivo, (k + 1) * por_archivo),
            "hospital_id": rng.integers(1, 41, n),
            "edad": np.round(edad, 1),
            "sexo": rng.choice(["F", "M"], n),
            "lactato": np.round(lactato, 2),
            "creatinina": np.round(creatinina, 2),
            "diuresis_24h": np.round(diuresis, 0),
            "apache_ii": np.round(apache, 0),
            "sofa": np.round(sofa, 0),
            "frec_cardiaca": np.round(frec_card, 0),
            "pa_media": np.round(pa_media, 0),
            "leucocitos": np.round(leucocitos, 2),
            "plaquetas": np.round(plaquetas, 0),
            "calcio": np.round(calcio, 2),
            "dias_uci": np.round(np.clip(rng.lognormal(1.1, 0.8, n), 0.2, 90), 1),
            "fallecido_uci": fallecido,
        })

        # Datos faltantes realistas: en UCI no todos los laboratorios se piden
        # a todos los pacientes.
        for col, tasa in [("lactato", 0.14), ("calcio", 0.09),
                          ("leucocitos", 0.05), ("diuresis_24h", 0.11)]:
            df.loc[df.sample(frac=tasa, random_state=SEMILLA + k).index, col] = np.nan

        df.to_csv(DIR_MED_RAW / f"cohorte_{k:02d}.csv", index=False)

    tam = sum(f.stat().st_size for f in DIR_MED_RAW.glob("*.csv")) / 1e6
    print(f"[04] {N_ARCHIVOS} archivos, {tam:.0f} MB")


def main():
    print("[04] CASO MEDICO - cohorte sintetica de UCI (sin datos identificables)")
    generar_cohorte()

    # --- Preprocesamiento con Dask (requisito tecnico de la guia) ---------
    t0 = time.time()
    ddf = dd.read_csv(str(DIR_MED_RAW / "cohorte_*.csv"), blocksize="16MB")
    print(f"[04] Particiones de Dask: {ddf.npartitions}")
    n = len(ddf)
    print(f"[04] Estancias: {n:,}")

    nulos = ddf.isna().sum().compute()
    print("[04] Nulos por variable:")
    print(nulos[nulos > 0].to_string())

    # Imputacion por mediana, calculada de forma distribuida
    for col in ["lactato", "calcio", "leucocitos", "diuresis_24h"]:
        mediana = ddf[col].quantile(0.5).compute()
        ddf[col] = ddf[col].fillna(mediana)

    # Variables derivadas con sentido clinico
    ddf["diuresis_por_kg_h"] = ddf["diuresis_24h"] / (70 * 24)   # peso estandar 70 kg
    ddf["indice_shock"] = ddf["frec_cardiaca"] / ddf["pa_media"]
    ddf["grupo_edad"] = ddf["edad"].map_partitions(
        lambda s: pd.cut(s, [0, 45, 65, 80, 120],
                         labels=["<45", "45-65", "65-80", "80+"]).astype(str))
    ddf["lactato_alto"] = (ddf["lactato"] > 2.0).astype(int)

    if DIR_MEDICO.exists():
        shutil.rmtree(DIR_MEDICO)
    ddf.to_parquet(str(DIR_MEDICO), engine="pyarrow", compression="snappy",
                   write_index=False)
    t_dask = time.time() - t0

    tam_csv = sum(f.stat().st_size for f in DIR_MED_RAW.glob("*.csv")) / 1e6
    tam_pq = sum(f.stat().st_size for f in DIR_MEDICO.rglob("*.parquet")) / 1e6
    print(f"[04] Preprocesamiento con Dask en {t_dask:.1f} s")
    print(f"[04] CSV {tam_csv:.0f} MB -> Parquet {tam_pq:.0f} MB "
          f"(reduccion {(1-tam_pq/tam_csv)*100:.0f}%)")
    registrar_metrica("preprocesamiento_medico", "4-caso-medico", "dask", n,
                      t_dask, medir_memoria(), f"CSV {tam_csv:.0f}MB -> Parquet {tam_pq:.0f}MB")

    # --- Analisis agregado, con Dask --------------------------------------
    t0 = time.time()
    por_grupo = (dd.read_parquet(str(DIR_MEDICO))
                 .groupby("grupo_edad")
                 .agg({"fallecido_uci": "mean", "apache_ii": "mean",
                       "lactato": "mean", "id_estancia": "count"})
                 .compute()
                 .rename(columns={"fallecido_uci": "mortalidad",
                                  "id_estancia": "estancias"}))
    por_grupo = por_grupo.sort_index()
    print(f"[04] Mortalidad por grupo de edad ({time.time()-t0:.1f} s):")
    print(por_grupo.round(3).to_string())
    por_grupo.to_csv(DIR_METRICAS / "medico_por_edad.csv")

    # Mortalidad segun lactato, la variable que el articulo senala como clave
    t0 = time.time()
    dfm = dd.read_parquet(str(DIR_MEDICO))
    dfm["bin_lactato"] = (dfm["lactato"] // 1).clip(upper=12)
    por_lactato = (dfm.groupby("bin_lactato")
                   .agg({"fallecido_uci": "mean", "id_estancia": "count"})
                   .compute().sort_index()
                   .rename(columns={"fallecido_uci": "mortalidad",
                                    "id_estancia": "estancias"}))
    por_lactato = por_lactato[por_lactato["estancias"] > 200]
    print(f"[04] Mortalidad segun lactato ({time.time()-t0:.1f} s):")
    print(por_lactato.round(3).to_string())
    por_lactato.to_csv(DIR_METRICAS / "medico_por_lactato.csv")

    # --- Modelo, replicando el esquema del articulo -----------------------
    #
    # El articulo usa XGBoost; aqui usamos GradientBoosting de scikit-learn, que
    # es el mismo tipo de modelo (potenciacion de gradiente sobre arboles) sin
    # anadir una dependencia mas al contenedor.
    t0 = time.time()
    muestra = dd.read_parquet(str(DIR_MEDICO)).sample(frac=0.25,
                                                      random_state=SEMILLA).compute()
    predictoras = ["edad", "lactato", "creatinina", "diuresis_24h", "apache_ii",
                   "sofa", "frec_cardiaca", "pa_media", "leucocitos",
                   "plaquetas", "calcio", "indice_shock", "diuresis_por_kg_h"]
    X = muestra[predictoras]
    y = muestra["fallecido_uci"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=SEMILLA, stratify=y)
    modelo = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                        random_state=SEMILLA)
    modelo.fit(X_tr, y_tr)
    prob = modelo.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, prob)
    t_modelo = time.time() - t0

    print(f"[04] Modelo entrenado en {t_modelo:.1f} s sobre {len(X_tr):,} estancias")
    print(f"[04] AUC en prueba: {auc:.4f}")
    print(classification_report(y_te, modelo.predict(X_te),
                                target_names=["sobrevive", "fallece"], digits=3))

    perm = permutation_importance(modelo, X_te, y_te, n_repeats=5,
                                  random_state=SEMILLA, scoring="roc_auc")
    imp = (pd.DataFrame({"variable": predictoras, "importancia": perm.importances_mean})
           .sort_values("importancia", ascending=False))
    print("[04] Variables mas influyentes (importancia por permutacion):")
    print(imp.head(8).to_string(index=False))
    imp.to_csv(DIR_METRICAS / "medico_importancia.csv", index=False)
    pd.DataFrame([{"auc": auc, "n_entrenamiento": len(X_tr),
                   "n_prueba": len(X_te),
                   "mortalidad_global": float(y.mean())}]).to_csv(
        DIR_METRICAS / "medico_metricas.csv", index=False)

    registrar_metrica("modelo_medico", "4-caso-medico", "sklearn", len(X_tr),
                      t_modelo, medir_memoria(), f"AUC={auc:.4f}")

    print("\n[04] CONTRASTE CON EL ARTICULO")
    print("[04] El articulo reporta lactato serico, APACHE II, diuresis total y")
    print("[04] calcio serico como las variables mas influyentes. Nuestro ranking")
    print("[04] sobre la cohorte sintetica es:")
    for _, fila in imp.head(4).iterrows():
        print(f"[04]   - {fila['variable']}")
    print("[04] La coincidencia era esperable: las relaciones se fijaron al")
    print("[04] generar los datos siguiendo lo que reporta la literatura. Lo que")
    print("[04] se valida aqui es el FLUJO METODOLOGICO, no el hallazgo clinico.")


if __name__ == "__main__":
    main()
