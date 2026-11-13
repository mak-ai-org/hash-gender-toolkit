# Base image: Python 3.11 slim
FROM python:3.11-slim

# Set environment variables for Python and Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set working directory inside container
WORKDIR /app

# Install curl for container health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and data files
COPY . .

# Expose default Streamlit web port
EXPOSE 8501

# Healthcheck to verify the PyShiny service is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/ || exit 1

# Launch the PyShiny application
ENTRYPOINT ["shiny", "run"]
CMD ["--host", "0.0.0.0", "--port", "8501", "app_shiny.py"]

