# One image for every Python service of the stack: MLflow server, REST API and Streamlit dashboard.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY streamlit_app ./streamlit_app

# Non-root user; the directories below are mount points of named volumes and inherit this ownership.
RUN useradd --create-home --uid 1000 fishpond \
    && mkdir -p /app/runtime /app/data /mlflow \
    && chown -R fishpond:fishpond /app /mlflow
USER fishpond

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=20 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "fishpond.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
