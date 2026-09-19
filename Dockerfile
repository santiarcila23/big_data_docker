FROM python:3.11-slim-bookworm

LABEL proyecto="practica-bigdata-dask-spark"
LABEL institucion="Institucion Universitaria de Envigado"
LABEL asignatura="Big Data"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PATH="/usr/lib/jvm/java-17-openjdk-amd64/bin:${PATH}"

# Dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps \
        curl \
        && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias de Python, se copia requirements.txt antes que el codigo para que Docker reutilice la capa de dependencias cuando solo cambia el codigo, ahorra minutos en cada reconstruccion
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo y datos
COPY src/ ./src/
COPY data/raw/ ./data/raw/
COPY README.md .

RUN mkdir -p data/interim outputs/figuras outputs/metricas

# Escala reducida por defecto dentro del contenedor 4 epocas (400.000 filas)
# Se puede cambiar al ejecutar docker run -e ESCALA=grande 
ENV ESCALA=pequena \
    SPARK_DRIVER_MEM=2g \
    SPARK_SHUFFLE_PARTITIONS=8

# Comando de inicio 
# Ejecuta el pipeline completo de punta a punta
CMD ["python", "src/run_pipeline.py"]
