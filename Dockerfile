FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
# Install to a specific directory to make copying easier
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final Runtime
FROM python:3.11-slim
WORKDIR /app
# Only copy the installed packages, not the pip cache or build tools
COPY --from=builder /install /usr/local
# BE SPECIFIC: Only copy your code files, not the whole folder
COPY main.py .
COPY services/ ./services/
COPY celery_config.py .
COPY worker.py .
# Ensure your .dockerignore excludes .venv/ and data/!

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]