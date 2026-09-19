"""
Etapa 6 - Visualizaciones
=========================

La guia exige un minimo de tres visualizaciones a partir de resultados agregados,
y advierte explicitamente que NO se deben convertir datos masivos completos a
pandas solo para graficar.

Este script cumple esa condicion por construccion: no toca el Parquet de 3
millones de filas. Lee unicamente los CSV de resultados ya agregados que
escribieron las etapas anteriores, ninguno de los cuales supera unos pocos miles
de filas. El trabajo pesado (binning, conteos, promedios) se hizo de forma
distribuida en Spark.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import DIR_METRICAS, DIR_FIGURAS

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold"})

AZUL, NARANJA, VERDE, ROJO, MORADO = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"
COLOR_CLASE = {"GALAXY": AZUL, "STAR": NARANJA, "QSO": VERDE}


def leer(nombre):
    ruta = DIR_METRICAS / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


def fig1_distribucion_redshift():
    df = leer("hist_redshift.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for clase, sub in df.groupby("class"):
        sub = sub.sort_values("bin")
        ax.plot(sub["bin"], sub["count"], lw=2, label=clase,
                color=COLOR_CLASE.get(clase, MORADO))
        ax.fill_between(sub["bin"], sub["count"], alpha=0.18,
                        color=COLOR_CLASE.get(clase, MORADO))
    ax.set_yscale("log")
    ax.set(xlabel="Corrimiento al rojo (redshift)",
           ylabel="Numero de observaciones (escala log)",
           title="Distribucion del redshift por tipo de objeto")
    ax.legend(title="Clase espectral")
    fig.text(0.5, -0.03,
             "Calculado en Spark sobre 2,9 M de observaciones; aqui solo se "
             "grafican los conteos por intervalo.",
             ha="center", fontsize=9, color="grey")
    plt.savefig(DIR_FIGURAS / "01_distribucion_redshift.png")
    plt.close()
    print("[06] 01_distribucion_redshift.png")


def fig2_diagrama_color():
    df = leer("diagrama_color.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    clases = ["GALAXY", "STAR", "QSO"]
    for ax, clase in zip(axes, clases):
        sub = df[df["class"] == clase]
        if sub.empty:
            continue
        sc = ax.scatter(sub["gr_bin"], sub["ri_bin"], c=sub["count"],
                        s=14, cmap="viridis", norm=matplotlib.colors.LogNorm())
        ax.set(title=clase, xlabel="color g - r")
        if ax is axes[0]:
            ax.set_ylabel("color r - i")
        plt.colorbar(sc, ax=ax, label="objetos")
    fig.suptitle("Diagrama color-color: cada tipo de objeto ocupa su propia region",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DIR_FIGURAS / "02_diagrama_color.png")
    plt.close()
    print("[06] 02_diagrama_color.png")


def fig3_comparacion_motores():
    df = leer("comparacion_dask_spark.csv")
    if df is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    for k, (tam, sub) in enumerate(df.groupby("tamano")):
        pos = np.arange(len(sub))
        ax[k].barh(pos - 0.2, sub["dask_seg"], height=0.38, color=NARANJA, label="Dask")
        ax[k].barh(pos + 0.2, sub["spark_seg"], height=0.38, color=AZUL, label="Spark")
        ax[k].set_yticks(pos)
        ax[k].set_yticklabels([o.split(".")[0] + ". " + o.split(". ")[1].split(" (")[0]
                               for o in sub["operacion"]], fontsize=9)
        ax[k].invert_yaxis()
        ax[k].set(xlabel="Segundos (menor es mejor)",
                  title=f"Tamano: {tam} del conjunto")
        ax[k].legend()
        for i, (d, s) in enumerate(zip(sub["dask_seg"], sub["spark_seg"])):
            ax[k].text(d + 0.3, i - 0.2, f"{d:.1f}", va="center", fontsize=8)
            ax[k].text(s + 0.3, i + 0.2, f"{s:.1f}", va="center", fontsize=8)

    fig.suptitle("Dask vs Spark sobre la misma operacion y los mismos datos",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DIR_FIGURAS / "03_comparacion_dask_spark.png")
    plt.close()
    print("[06] 03_comparacion_dask_spark.png")


def fig4_importancia_mllib():
    df = leer("importancia_mllib.csv")
    met = leer("metricas_mllib.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    df = df.sort_values("importancia")
    colores = [VERDE if v > 0.15 else AZUL for v in df["importancia"]]
    ax.barh(df["variable"], df["importancia"], color=colores)
    for i, v in enumerate(df["importancia"]):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9)
    titulo = "Importancia de variables - RandomForest de MLlib"
    if met is not None:
        titulo += f"  (exactitud {met['exactitud'].iloc[0]:.1%})"
    ax.set(xlabel="Importancia", title=titulo, xlim=(0, df["importancia"].max() * 1.18))
    plt.savefig(DIR_FIGURAS / "04_importancia_mllib.png")
    plt.close()
    print("[06] 04_importancia_mllib.png")


def fig5_algoritmo_genetico():
    conv = leer("ag_convergencia.csv")
    esc = leer("ag_escalabilidad.csv")
    if conv is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(conv["generacion"], conv["mejor"], lw=2, color=VERDE, label="Mejor individuo")
    ax[0].plot(conv["generacion"], conv["media"], lw=1.6, color=NARANJA,
               label="Media de la poblacion")
    ax[0].set(xlabel="Generacion", ylabel="Aptitud",
              title="Convergencia del algoritmo genetico")
    ax[0].legend()

    if esc is not None:
        ax[1].plot(esc["trabajadores"], esc["aceleracion"], marker="o", lw=2.2,
                   color=AZUL, label="Aceleracion medida")
        ax[1].plot(esc["trabajadores"], esc["trabajadores"], "--", color="grey",
                   label="Aceleracion ideal")
        ax[1].axhline(1.0, color=ROJO, ls=":", label="Sin ganancia")
        ax[1].set(xlabel="Trabajadores de Dask", ylabel="Aceleracion (x)",
                  title="Escalabilidad en una maquina de 1 nucleo")
        ax[1].legend()
    plt.tight_layout()
    plt.savefig(DIR_FIGURAS / "05_algoritmo_genetico.png")
    plt.close()
    print("[06] 05_algoritmo_genetico.png")


def fig6_caso_medico():
    lac = leer("medico_por_lactato.csv")
    imp = leer("medico_importancia.csv")
    if lac is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(lac["bin_lactato"], lac["mortalidad"] * 100, marker="o", lw=2.2,
               color=ROJO)
    ax[0].set(xlabel="Lactato serico (mmol/L)", ylabel="Mortalidad en UCI (%)",
              title="Mortalidad segun lactato (cohorte sintetica)")

    if imp is not None:
        top = imp.head(8).sort_values("importancia")
        ax[1].barh(top["variable"], top["importancia"], color=MORADO)
        ax[1].set(xlabel="Caida del AUC al permutar la variable",
                  title="Variables mas influyentes")
    plt.tight_layout()
    plt.savefig(DIR_FIGURAS / "06_caso_medico.png")
    plt.close()
    print("[06] 06_caso_medico.png")


def fig7_calidad_campos():
    cal = leer("calidad_vs_fotometria.csv")
    reg = leer("por_region.csv")
    if cal is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    colores = {"buena": VERDE, "media": NARANJA, "pobre": ROJO}
    ax[0].bar(cal["calidad"], cal["objetos"] / 1000,
              color=[colores.get(c, AZUL) for c in cal["calidad"]])
    for i, (v, s) in enumerate(zip(cal["objetos"] / 1000, cal["seeing_medio"])):
        ax[0].text(i, v + 20, f"seeing {s:.2f}\"", ha="center", fontsize=9)
    ax[0].set(ylabel="Observaciones (miles)",
              title="Resultado del join: calidad del campo observacional")

    if reg is not None:
        r = reg.sort_values("objetos", ascending=False).head(10)
        ax[1].bar(r["region"].astype(str), r["objetos"] / 1000, color=AZUL)
        ax[1].set(xlabel="Region del cielo", ylabel="Observaciones (miles)",
                  title="Reparto por particion espacial")
        ax[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(DIR_FIGURAS / "07_calidad_y_regiones.png")
    plt.close()
    print("[06] 07_calidad_y_regiones.png")


def main():
    print("[06] Generando visualizaciones a partir de resultados AGREGADOS")
    for f in (fig1_distribucion_redshift, fig2_diagrama_color,
              fig3_comparacion_motores, fig4_importancia_mllib,
              fig5_algoritmo_genetico, fig6_caso_medico, fig7_calidad_campos):
        try:
            f()
        except Exception as e:
            print(f"[06] AVISO: {f.__name__} fallo ({e})")
    n = len(list(DIR_FIGURAS.glob("*.png")))
    print(f"[06] {n} figuras en {DIR_FIGURAS}")


if __name__ == "__main__":
    main()
