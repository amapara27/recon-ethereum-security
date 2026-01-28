# Docker Deployment Guide

This guide explains how to deploy Recon using Docker and Docker Compose.

## Quick Start

```bash
# 1. Configure environment variables
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys

# 2. Start the application
docker-compose up --build

# 3. Access the application
# - API Docs: http://localhost:8000/docs
# - Fraud Alerts: http://localhost:8000/alerts
# - Contract Analysis: POST http://localhost:8000/analyze-contract
```

## Architecture

The Docker setup includes:

### Services
- **backend**: FastAPI server + Blockchain monitor

### Volumes (Persistent Data)
- `alerts.db`: Fraud detection alerts database
- `contract_cache.db`: Smart contract analysis cache (reduces API costs)
- `models/`: Pre-trained ML models
- `src/lists/`: Address whitelists/blacklists

### Environment Variables
Required in `backend/.env`:
- `INFURA_RPC_URL`: Ethereum RPC endpoint
- `ETHERSCAN_API_KEY`: For transaction history
- `ANTHROPIC_API_KEY`: For AI-powered contract analysis

## Commands

### Start Services
```bash
# Start in foreground (see logs)
docker-compose up

# Start in background
docker-compose up -d

# Rebuild and start
docker-compose up --build
```

### View Logs
```bash
# All logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs -f backend
```

### Stop Services
```bash
# Stop (preserves containers)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove everything including volumes
docker-compose down -v
```

### Health Check
```bash
# Check service status
docker-compose ps

# Check health
curl http://localhost:8000/docs
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs backend

# Verify environment variables
docker-compose config

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Database issues
```bash
# Reset databases (WARNING: deletes all data)
rm backend/alerts.db backend/contract_cache.db
docker-compose restart
```

### API not responding
```bash
# Check if container is running
docker-compose ps

# Check container health
docker inspect recon-backend | grep Health

# Restart service
docker-compose restart backend
```

## Production Deployment

For production (AWS EC2, etc.):

1. Use environment-specific `.env` files
2. Set up proper networking and security groups
3. Configure SSL/TLS with reverse proxy (nginx)
4. Set up monitoring and log aggregation
5. Use Docker secrets for sensitive data
6. Configure automatic backups for database volumes

Example production command:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Monitoring

### Check resource usage
```bash
docker stats
````markdown
# Docker Deployment Guide

This guide explains how to deploy Recon using Docker and Docker Compose.

## Quick Start

```bash
# 1. Configure environment variables
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys

# Ensure local directories and files exist so volumes mount correctly
mkdir -p backend/database
touch backend/database/alerts.db backend/database/contract_cache.db
mkdir -p backend/app/models

# 2. Start the application
docker-compose up --build

# 3. Access the application
# - API Docs: http://localhost:8000/docs
# - Fraud Alerts: http://localhost:8000/alerts
# - Contract Analysis: POST http://localhost:8000/analyze-contract
```

## Architecture

The Docker setup includes:

### Services
- **backend**: FastAPI server + Blockchain monitor

### Volumes (Persistent Data)
- `alerts.db`: Fraud detection alerts database
- `contract_cache.db`: Smart contract analysis cache (reduces API costs)
- `models/`: Pre-trained ML models
- `src/lists/`: Address whitelists/blacklists

Notes on paths used by `docker-compose`:
- Local database files are stored under `./backend/database/` and are mounted into the container at `/app/backend/database/`.
- Local model artifacts should be placed in `./backend/app/models/` and are mounted into the container at `/app/backend/app/models/`.
- Address lists live in `./backend/app/config/lists/` and are mounted into the container at `/app/backend/app/config/lists/`.

### Environment Variables
Required in `backend/.env`:
- `INFURA_RPC_URL` or `ALCHEMY_RPC_URL`: Ethereum RPC endpoint
- `ETHERSCAN_API_KEY`: For transaction history
- `ANTHROPIC_API_KEY`: For AI-powered contract analysis

## Commands

### Start Services
```bash
# Start in foreground (see logs)
docker-compose up

# Start in background
docker-compose up -d

# Rebuild and start
docker-compose up --build
```

### View Logs
```bash
# All logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs -f backend
```

### Stop Services
```bash
# Stop (preserves containers)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove everything including volumes
docker-compose down -v
```

### Health Check
```bash
# Check service status
docker-compose ps

# Check health
curl http://localhost:8000/docs
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs backend

# Verify environment variables
docker-compose config

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Database issues
```bash
# Reset databases (WARNING: deletes all data)
rm backend/database/alerts.db backend/database/contract_cache.db
docker-compose restart
```

### API not responding
```bash
# Check if container is running
docker-compose ps

# Check container health
docker inspect recon-backend | grep Health

# Restart service
docker-compose restart backend
```

## Production Deployment

For production (AWS EC2, etc.):

1. Use environment-specific `.env` files
2. Set up proper networking and security groups
3. Configure SSL/TLS with reverse proxy (nginx)
4. Set up monitoring and log aggregation
5. Use Docker secrets for sensitive data
6. Configure automatic backups for database volumes

Note: the container's startup script launches both the blockchain `monitor` service and the FastAPI `api` service and is configured to wait for both processes. For more robust per-process supervision, logging, and auto-restart behavior consider running the monitor as a separate Compose service or using a process manager like `supervisord` in production.

Example production command:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Monitoring

### Check resource usage
```bash
docker stats
```

### Check database size
```bash
ls -lh backend/database/*.db
```

### Check contract cache efficiency
```bash
# Count cached contracts
sqlite3 backend/database/contract_cache.db "SELECT COUNT(*) FROM contract_analyses;"
```

## Updating

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose up --build -d

# View new logs
docker-compose logs -f
```

````
