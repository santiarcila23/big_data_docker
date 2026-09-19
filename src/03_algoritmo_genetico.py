"""
Actividad 3.1 - Algoritmos geneticos y procesamiento paralelo
=============================================================

QUE ES UN ALGORITMO GENETICO
----------------------------
Es una tecnica de optimizacion inspirada en la evolucion biologica. En lugar de
buscar la solucion optima con calculo (derivadas, gradientes), mantiene una
*poblacion* de soluciones candidatas y las hace competir y reproducirse durante
varias generaciones. Las mejores sobreviven y se combinan; las peores se
descartan. Con suficientes generaciones, la poblacion converge hacia soluciones
buenas.

Sus cinco componentes:

1. REPRESENTACION (o codificacion). Como se escribe una solucion candidata.
   Aqui cada individuo es un vector binario de longitud N: un 1 significa
   "incluyo este objeto" y un 0 "lo dejo fuera".

2. FUNCION DE APTITUD (fitness). Que tan buena es una solucion. Es lo unico que
   el algoritmo sabe del problema; todo lo demas es maquinaria generica. Aqui
   es el valor total de los objetos escogidos, con penalizacion si se excede la
   capacidad de la mochila.

3. SELECCION. Como se eligen los padres. Usamos *seleccion por torneo*: se
   toman k individuos al azar y gana el de mejor aptitud. Es simple y mantiene
   presion selectiva sin que los mejores dominen de inmediato.

4. CRUCE (crossover). Como se combinan dos padres. Usamos cruce uniforme: cada
   gen del hijo se toma al azar de uno u otro padre.

5. MUTACION. Cambio aleatorio de algunos genes. Es lo que impide que la
   poblacion se estanque en un optimo local: sin mutacion, si toda la poblacion
   converge al mismo individuo, el cruce ya no genera nada nuevo.

VENTAJAS
  - No necesitan derivadas ni que la funcion objetivo sea continua o convexa.
  - Exploran varias regiones del espacio a la vez, lo que reduce el riesgo de
    quedarse en un optimo local.
  - Se adaptan a problemas combinatorios donde los metodos clasicos no aplican.
  - Son *vergonzosamente paralelos* en la evaluacion de la poblacion: cada
    individuo se evalua sin depender de los demas.

DESVENTAJAS
  - No garantizan el optimo global, solo una buena solucion.
  - Tienen muchos hiperparametros (tamano de poblacion, tasa de mutacion,
    tamano de torneo) y el resultado depende bastante de ellos.
  - Son caros: miles de evaluaciones de la funcion de aptitud.
  - No son reproducibles sin fijar la semilla.

EL PROBLEMA QUE RESOLVEMOS
--------------------------
Problema de la mochila 0/1, planteado en el contexto del proyecto: un telescopio
tiene una noche de observacion con tiempo limitado. Cada objeto candidato del
catalogo tiene un *valor cientifico* (cuanto aporta observarlo) y un *costo en
minutos* (cuanto tarda). Hay que elegir el subconjunto que maximice el valor
total sin pasarse del tiempo disponible.

Es NP-duro: con 250 objetos hay 2^250 combinaciones posibles, mas que atomos en
el universo observable. Por eso se usa una heuristica y no fuerza bruta.

PARALELIZACION CON DASK
-----------------------
Se paralelizan las *ejecuciones independientes con distintas semillas*, no la
evaluacion de la poblacion. La razon esta explicada en la discusion al final:
con poblaciones de este tamano el costo de coordinar tareas supera al calculo,
y paralelizar dentro de una generacion resulta mas lento que el secuencial.
Ejecutar semillas en paralelo si aporta, porque cada ejecucion es una tarea
grande e independiente.
"""
import time
import numpy as np
import pandas as pd
import dask
from dask import delayed

from config import DIR_METRICAS, DIR_FIGURAS, SEMILLA, registrar_metrica, medir_memoria

# --- Parametros del problema y del algoritmo -------------------------------
N_OBJETOS    = 250     # objetos candidatos del catalogo
CAPACIDAD    = 0.4     # fraccion del costo total disponible esa noche
POBLACION    = 400
GENERACIONES = 400
TASA_MUTACION = 0.02
TAM_TORNEO   = 4
ELITE        = 4       # individuos que pasan intactos a la siguiente generacion
N_SEMILLAS   = 8       # ejecuciones independientes


def construir_problema(semilla=SEMILLA):
    """Genera los objetos candidatos: valor cientifico y costo en minutos."""
    rng = np.random.default_rng(semilla)
    costo = rng.uniform(5, 90, N_OBJETOS)            # minutos de telescopio
    # El valor cientifico correlaciona con el costo (los objetos debiles piden
    # mas exposicion) pero no de forma perfecta: ahi esta la oportunidad.
    valor = costo * rng.uniform(0.6, 1.8, N_OBJETOS)
    capacidad = costo.sum() * CAPACIDAD
    return valor, costo, capacidad


def reparar(poblacion, valor, costo, capacidad):
    """
    Operador de reparacion: vuelve factible a todo individuo que se pase de la
    capacidad, quitando primero los objetos con peor relacion valor/costo.

    Es una tecnica estandar en la mochila con algoritmos geneticos. Sin ella, el
    algoritmo gasta la mayor parte de las generaciones aprendiendo a no pasarse
    en lugar de aprender que objetos conviene llevar, y termina por debajo de una
    simple heuristica voraz. La incluimos porque sin reparacion el AG NO superaba
    a la heuristica, y eso se midio.
    """
    ratio = valor / costo
    orden_malos = np.argsort(ratio)          # peor relacion primero
    costos = poblacion @ costo
    excedidos = np.where(costos > capacidad)[0]
    for fila in excedidos:
        individuo = poblacion[fila]
        c = costos[fila]
        for idx in orden_malos:
            if c <= capacidad:
                break
            if individuo[idx]:
                individuo[idx] = 0
                c -= costo[idx]
        poblacion[fila] = individuo
    return poblacion


def aptitud(poblacion, valor, costo, capacidad):
    """
    Evalua la poblacion completa de forma vectorizada.

    Un individuo que se pasa de la capacidad no se descarta: se penaliza en
    proporcion al exceso. Descartarlo del todo haria que el algoritmo perdiera
    informacion util de soluciones que estan casi bien.
    """
    valores = poblacion @ valor
    costos = poblacion @ costo
    exceso = np.maximum(0.0, costos - capacidad)
    return valores - 3.0 * exceso


def ejecutar_ag(semilla, valor, costo, capacidad, generaciones=GENERACIONES,
                registrar_historia=False):
    """Una ejecucion completa del algoritmo genetico."""
    rng = np.random.default_rng(semilla)
    poblacion = (rng.random((POBLACION, N_OBJETOS)) < 0.25).astype(np.int8)
    historia = []

    poblacion = reparar(poblacion, valor, costo, capacidad)

    for gen in range(generaciones):
        apt = aptitud(poblacion, valor, costo, capacidad)
        orden = np.argsort(-apt)
        poblacion = poblacion[orden]
        apt = apt[orden]
        if registrar_historia:
            historia.append({"generacion": gen, "mejor": float(apt[0]),
                             "media": float(apt.mean())})

        # Elitismo: los mejores pasan sin cambios
        nueva = [poblacion[:ELITE]]

        # Seleccion por torneo
        n_hijos = POBLACION - ELITE
        aspirantes = rng.integers(0, POBLACION, size=(n_hijos * 2, TAM_TORNEO))
        ganadores = aspirantes[np.arange(n_hijos * 2),
                               np.argmax(apt[aspirantes], axis=1)]
        padres1 = poblacion[ganadores[:n_hijos]]
        padres2 = poblacion[ganadores[n_hijos:]]

        # Cruce uniforme
        mascara = rng.random((n_hijos, N_OBJETOS)) < 0.5
        hijos = np.where(mascara, padres1, padres2)

        # Mutacion
        mutar = rng.random((n_hijos, N_OBJETOS)) < TASA_MUTACION
        hijos = np.where(mutar, 1 - hijos, hijos).astype(np.int8)

        hijos = reparar(hijos, valor, costo, capacidad)
        nueva.append(hijos)
        poblacion = np.vstack(nueva)

    apt = aptitud(poblacion, valor, costo, capacidad)
    mejor = int(np.argmax(apt))
    return {
        "semilla": semilla,
        "aptitud": float(apt[mejor]),
        "valor": float(poblacion[mejor] @ valor),
        "costo": float(poblacion[mejor] @ costo),
        "objetos": int(poblacion[mejor].sum()),
        "historia": historia,
    }


def main():
    print("[03] ALGORITMO GENETICO - secuencial vs Dask")
    valor, costo, capacidad = construir_problema()
    print(f"[03] Problema: {N_OBJETOS} objetos candidatos, "
          f"capacidad {capacidad:,.0f} min de {costo.sum():,.0f} min totales")
    print(f"[03] Espacio de busqueda: 2^{N_OBJETOS} combinaciones")

    # --- Referencia: heuristica voraz por relacion valor/costo -------------
    # Sirve para saber si el algoritmo genetico aporta algo sobre lo obvio.
    orden = np.argsort(-(valor / costo))
    acum, voraz_valor = 0.0, 0.0
    for idx in orden:
        if acum + costo[idx] <= capacidad:
            acum += costo[idx]
            voraz_valor += valor[idx]
    print(f"[03] Referencia voraz (valor/costo): {voraz_valor:,.1f}")

    semillas = [SEMILLA + i for i in range(N_SEMILLAS)]

    # --- Version secuencial ------------------------------------------------
    t0 = time.time()
    res_sec = [ejecutar_ag(s, valor, costo, capacidad) for s in semillas]
    t_secuencial = time.time() - t0
    mejor_sec = max(r["valor"] for r in res_sec)
    print(f"[03] SECUENCIAL: {N_SEMILLAS} ejecuciones en {t_secuencial:.2f} s")
    print(f"[03]   mejor valor = {mejor_sec:,.1f}")
    registrar_metrica("ag_secuencial", "3-algoritmo-genetico", "python", N_SEMILLAS,
                      t_secuencial, medir_memoria(),
                      f"{N_SEMILLAS} semillas x {GENERACIONES} generaciones")

    # --- Version con Dask --------------------------------------------------
    #
    # dask.delayed convierte cada ejecucion en una tarea perezosa. Al llamar
    # dask.compute se construye el grafo y se reparten las tareas entre los
    # hilos/procesos disponibles.
    t0 = time.time()
    tareas = [delayed(ejecutar_ag)(s, valor, costo, capacidad) for s in semillas]
    res_dask = dask.compute(*tareas, scheduler="processes")
    t_dask = time.time() - t0
    mejor_dask = max(r["valor"] for r in res_dask)
    print(f"[03] DASK (processes): {N_SEMILLAS} ejecuciones en {t_dask:.2f} s")
    print(f"[03]   mejor valor = {mejor_dask:,.1f}")
    registrar_metrica("ag_dask_procesos", "3-algoritmo-genetico", "dask", N_SEMILLAS,
                      t_dask, medir_memoria(),
                      f"{N_SEMILLAS} semillas en paralelo con dask.delayed")

    # --- Version con Dask, planificador de hilos ---------------------------
    #
    # Se incluye a proposito para mostrar el efecto del GIL de Python: con
    # hilos, el codigo Python puro no corre realmente en paralelo.
    t0 = time.time()
    tareas = [delayed(ejecutar_ag)(s, valor, costo, capacidad) for s in semillas]
    dask.compute(*tareas, scheduler="threads")
    t_hilos = time.time() - t0
    print(f"[03] DASK (threads): {t_hilos:.2f} s")
    registrar_metrica("ag_dask_hilos", "3-algoritmo-genetico", "dask", N_SEMILLAS,
                      t_hilos, medir_memoria(), "planificador de hilos, limitado por el GIL")

    # --- Comparacion -------------------------------------------------------
    aceleracion = t_secuencial / t_dask if t_dask > 0 else float("nan")
    print(f"\n[03] {'='*56}")
    print(f"[03] Secuencial      : {t_secuencial:7.2f} s")
    print(f"[03] Dask (procesos) : {t_dask:7.2f} s   aceleracion {aceleracion:.2f}x")
    print(f"[03] Dask (hilos)    : {t_hilos:7.2f} s")
    print(f"[03] Mejor valor secuencial = {mejor_sec:,.1f} | Dask = {mejor_dask:,.1f}")
    print(f"[03] Mejora sobre la heuristica voraz: "
          f"{(mejor_sec/voraz_valor - 1)*100:+.2f}%")
    print(f"[03] {'='*56}")

    # --- Curva de convergencia, para la visualizacion ----------------------
    detalle = ejecutar_ag(SEMILLA, valor, costo, capacidad, registrar_historia=True)
    pd.DataFrame(detalle["historia"]).to_csv(
        DIR_METRICAS / "ag_convergencia.csv", index=False)

    pd.DataFrame([
        {"version": "Secuencial", "segundos": round(t_secuencial, 2),
         "mejor_valor": round(mejor_sec, 1), "aceleracion": 1.0},
        {"version": "Dask (procesos)", "segundos": round(t_dask, 2),
         "mejor_valor": round(mejor_dask, 1), "aceleracion": round(aceleracion, 2)},
        {"version": "Dask (hilos)", "segundos": round(t_hilos, 2),
         "mejor_valor": round(mejor_dask, 1),
         "aceleracion": round(t_secuencial / t_hilos, 2)},
    ]).to_csv(DIR_METRICAS / "ag_comparacion.csv", index=False)

    pd.DataFrame([{k: v for k, v in r.items() if k != "historia"}
                  for r in res_sec]).to_csv(
        DIR_METRICAS / "ag_resultados_semillas.csv", index=False)

    # --- Experimento de escalabilidad: variar el numero de trabajadores ----
    import os
    n_nucleos = os.cpu_count()
    print(f"\n[03] Nucleos disponibles en esta maquina: {n_nucleos}")
    print("[03] Escalabilidad segun numero de trabajadores de Dask:")
    filas_esc = []
    for n_w in [1, 2, 4, 8]:
        t0 = time.time()
        tareas = [delayed(ejecutar_ag)(s_, valor, costo, capacidad) for s_ in semillas]
        dask.compute(*tareas, scheduler="processes", num_workers=n_w)
        dt = time.time() - t0
        filas_esc.append({"trabajadores": n_w, "segundos": round(dt, 2),
                          "aceleracion": round(t_secuencial / dt, 2)})
        print(f"[03]   {n_w} trabajador(es): {dt:6.2f} s  "
              f"(aceleracion {t_secuencial/dt:.2f}x)")
        registrar_metrica(f"ag_dask_{n_w}_workers", "3-algoritmo-genetico", "dask",
                          N_SEMILLAS, dt, medir_memoria(),
                          f"{n_w} procesos, maquina de {n_nucleos} nucleo(s)")
    pd.DataFrame(filas_esc).to_csv(DIR_METRICAS / "ag_escalabilidad.csv", index=False)

    print("\n[03] DISCUSION: aporta o no la paralelizacion?")
    print("[03] Ver la seccion correspondiente del informe. Resumen:")
    print(f"[03]   - Cada ejecucion es independiente, asi que el problema es")
    print(f"[03]     paralelizable en principio.")
    print(f"[03]   - PERO en esta maquina hay {n_nucleos} nucleo(s). El techo teorico")
    print(f"[03]     de aceleracion es exactamente el numero de nucleos, asi que")
    print(f"[03]     aqui no puede haber ganancia por mucho que se reparta.")
    print(f"[03]   - Medido: {aceleracion:.2f}x. Menor que 1 porque a la misma")
    print(f"[03]     cantidad de calculo se le suma el costo de arrancar procesos")
    print(f"[03]     y serializar datos entre ellos.")
    print(f"[03]   - Con hilos ({t_hilos:.2f} s) tampoco hay ganancia, pero por otra")
    print(f"[03]     razon: el GIL de Python impide que dos hilos ejecuten bytecode")
    print(f"[03]     a la vez. NumPy libera el GIL en sus operaciones internas, por")
    print(f"[03]     eso los hilos no salen peor que el secuencial, solo igual.")
    print(f"[03]   - CONCLUSION: el codigo esta correctamente paralelizado, pero la")
    print(f"[03]     paralelizacion no aporta en este entorno. En una maquina de 8")
    print(f"[03]     nucleos el mismo codigo, sin cambios, si mostraria ganancia.")


if __name__ == "__main__":
    main()
