# 1. Usar una imagen base oficial de Python
FROM python:3.9-slim

# 2. Establecer un directorio de trabajo general (NO /app)
WORKDIR /project

# 3. Copiar el archivo de dependencias y instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el código de la aplicación a una subcarpeta 'app'
COPY ./app /project/app

# 5. Exponer el puerto en el que se ejecutará la API
EXPOSE 8000

# 6. Comando para ejecutar la aplicación desde el directorio /project
# Ahora uvicorn puede encontrar el módulo 'app' correctamente.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
