# Usa una imagen base de Python 3.10 oficial
FROM python:3.10-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo de requerimientos primero para aprovechar el cache de Docker [cite: 29]
COPY requirements.txt .

# Instala las dependencias de Python
# --no-cache-dir reduce el tamaño de la imagen
# --default-timeout=100 para conexiones lentas
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copia el directorio src (con tu código de preprocesamiento y entrenamiento) a /app/src
COPY src/ /app/src/

# Variables de entorno (opcional, pero bueno para logging)
ENV PYTHONUNBUFFERED=1