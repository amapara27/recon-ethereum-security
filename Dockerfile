# Dockerfile for Ethereum Fraud Detector Backend

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Create necessary directories
RUN mkdir -p backend/models backend/data

# Create startup script to run both API and monitor
RUN echo '#!/bin/bash\n\
python backend/src/monitor.py &\n\
python backend/src/api.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port for API
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run both the monitor and API
CMD ["/app/start.sh"]