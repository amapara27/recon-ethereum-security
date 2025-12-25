# Dockerfile for Recon - Ethereum Fraud Detection & Smart Contract Auditor

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for healthchecks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Create necessary directories for databases and models
RUN mkdir -p backend/models backend/data && \
    touch backend/alerts.db backend/contract_cache.db

# Create startup script to run both monitor and API server
RUN echo '#!/bin/bash\n\
set -e\n\
python backend/src/monitor.py &\n\
MONITOR_PID=$!\n\
python backend/src/api.py &\n\
API_PID=$!\n\
wait $API_PID\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port for FastAPI
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check to ensure API is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run both the monitor and API
CMD ["/app/start.sh"]
