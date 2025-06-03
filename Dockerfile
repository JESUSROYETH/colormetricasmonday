# Usa una imagen base de Python 3.10 (considera -slim para producción)
FROM python:3.10

# Establece variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Establece el directorio de trabajo
WORKDIR /app

# Instala dependencias del sistema necesarias para OpenCV y otras bibliotecas
# libgl1 y libglib2.0-0 son dependencias comunes para OpenCV.
# Si usas opencv-python-headless, podrías intentar omitir libgl1-mesa-glx
# y solo mantener libglib2.0-0, o añadir otras si son necesarias (ej. libjpeg-dev, libpng-dev).
# Para opencv-python (no headless), estas son generalmente buenas.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia el archivo de requerimientos primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instala las dependencias de Python
# Asegúrate que gunicorn esté en tu requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY . .

# Establece el puerto en el que la aplicación escuchará
# Cloud Run proporciona la variable PORT, Gunicorn la usará.
ENV PORT 8080
EXPOSE 8080

# Comando para ejecutar la aplicación Flask con Gunicorn
# Reemplaza 'main:app' con 'tu_archivo_python:tu_objeto_flask' si son diferentes.
# --workers: ajusta según los recursos de tu instancia de Cloud Run y la naturaleza de tu app.
# --threads: útil si tu código es I/O bound o usa bibliotecas que liberan el GIL.
# --timeout 0: para procesos potencialmente largos como el procesamiento de video. Considera un valor alto en lugar de 0 si es posible.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "0", "main:app"]