# Installation and Setup Guide

This guide provides detailed instructions for setting up NanoChemGPT in different environments, from development to production deployment.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Production Deployment](#production-deployment)
4. [Docker Installation](#docker-installation)
5. [Environment Configuration](#environment-configuration)
6. [Data Setup](#data-setup)
7. [Testing Installation](#testing-installation)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.11 or higher
- 8 GB RAM
- 10 GB free disk space
- Internet connection for API access and literature mining

**Recommended Requirements:**
- Python 3.11+
- 16 GB RAM
- 50 GB free disk space
- GPU support (optional, for faster embeddings)

### External Dependencies

**Required:**
- OpenAI API key (for embeddings and language model)

**Optional:**
- MongoDB (for persistent storage and job management)
- Redis (for caching and session management)
- Docker (for containerized deployment)

### API Keys Setup

1. **OpenAI API Key**:
   - Visit https://platform.openai.com/api-keys
   - Create a new API key
   - Ensure sufficient credits for embeddings and completions

2. **EU-PMC Access** (optional):
   - Register at https://europepmc.org/
   - No API key required for basic access

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/DMCarnahan/NanoChemGPT.git
cd NanoChemGPT
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n nanochemgpt python=3.11
conda activate nanochemgpt
```

### 3. Install Dependencies

```bash
# Install PyTorch CPU version first (recommended for compatibility)
pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install spaCy model
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Install all other requirements
pip install -r requirements.txt
```

**Note**: If you encounter dependency conflicts, try installing in this order:
1. PyTorch (CPU version)
2. spaCy and models
3. Remaining requirements

### 4. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env with your configuration
# Required variables:
nano .env  # or use your preferred editor
```

**Minimal .env configuration:**
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMB=text-embedding-3-small

# Embedding Backend
EMBED_BACKEND=st  # or 'openai' for OpenAI embeddings

# Application Settings
FLASK_ENV=development
DEBUG=True
PORT=5000

# Vector Store Paths (will be created automatically)
RETRIEVER_INDEX_DIRS=retriever/index_doc,retriever/index_passage
UPLOADS_DIR=data/uploads

# spaCy Model Path
SPACY_MODEL=harvester/miner/ner_model/model-best
```

### 5. Initialize Data Directories

```bash
# Create necessary directories
mkdir -p data/uploads
mkdir -p retriever/index_doc
mkdir -p retriever/index_passage
mkdir -p harvester/out_auto
mkdir -p logs
```

### 6. Download NER Model

If the custom spaCy NER model is not included in the repository:

```bash
# Option 1: Download from releases (if available)
wget https://github.com/DMCarnahan/NanoChemGPT/releases/download/v1.0.0/ner_model.tar.gz
tar -xzf ner_model.tar.gz -C harvester/miner/

# Option 2: Use default English model as fallback
# The system will automatically fall back to en_core_web_sm if custom model is not found
```

### 7. Test Installation

```bash
# Test basic imports
python -c "import app; print('Installation successful!')"

# Test with minimal functionality
python -c "
from app import app
with app.test_client() as client:
    response = client.get('/health')
    print(f'Health check: {response.status_code}')
"
```

### 8. Run Development Server

```bash
# Start Flask development server
python app.py

# Or use gunicorn for testing production setup
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

Access the application at `http://localhost:5000`

## Production Deployment

### 1. System Preparation

**Update system packages:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential

# CentOS/RHEL
sudo yum update -y
sudo yum install -y python311 python311-devel gcc gcc-c++
```

**Create application user:**
```bash
sudo useradd -m -s /bin/bash nanochemgpt
sudo usermod -aG sudo nanochemgpt
```

### 2. Application Setup

```bash
# Switch to application user
sudo su - nanochemgpt

# Clone and setup application
git clone https://github.com/DMCarnahan/NanoChemGPT.git
cd NanoChemGPT

# Create production virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Production-specific packages
pip install gunicorn supervisor nginx
```

### 3. Production Configuration

**Create production .env:**
```bash
# Production environment configuration
FLASK_ENV=production
DEBUG=False
PORT=8000

# Security settings
SECRET_KEY=your_secure_secret_key_here

# Database connections
MONGODB_URI=mongodb://localhost:27017/nanochemgpt
REDIS_URL=redis://localhost:6379/0

# Performance settings
WORKERS=4
WORKER_CONNECTIONS=1000
MAX_REQUESTS=1000
TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/nanochemgpt/app.log

# Rate limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

### 4. Database Setup (Optional)

**MongoDB:**
```bash
# Install MongoDB
sudo apt install -y mongodb-org

# Start and enable MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Create database user
mongo
> use nanochemgpt
> db.createUser({
    user: "nanochemgpt",
    pwd: "secure_password",
    roles: ["readWrite"]
  })
```

**Redis:**
```bash
# Install Redis
sudo apt install -y redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
# Set: maxmemory 1gb
# Set: maxmemory-policy allkeys-lru

# Restart Redis
sudo systemctl restart redis-server
```

### 5. Web Server Configuration

**Nginx configuration:**
```bash
sudo nano /etc/nginx/sites-available/nanochemgpt
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }

    location /static/ {
        alias /home/nanochemgpt/NanoChemGPT/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/nanochemgpt /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. Process Management

**Systemd service:**
```bash
sudo nano /etc/systemd/system/nanochemgpt.service
```

```ini
[Unit]
Description=NanoChemGPT Application
After=network.target

[Service]
Type=exec
User=nanochemgpt
Group=nanochemgpt
WorkingDirectory=/home/nanochemgpt/NanoChemGPT
Environment=PATH=/home/nanochemgpt/NanoChemGPT/venv/bin
EnvironmentFile=/home/nanochemgpt/NanoChemGPT/.env
ExecStart=/home/nanochemgpt/NanoChemGPT/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker main_asgi:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable nanochemgpt
sudo systemctl start nanochemgpt
sudo systemctl status nanochemgpt
```

### 7. SSL/HTTPS Setup

**Using Certbot (Let's Encrypt):**
```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Docker Installation

### 1. Using Docker Compose (Recommended)

**Create docker-compose.yml:**
```yaml
version: '3.8'

services:
  nanochemgpt:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MONGODB_URI=mongodb://mongo:27017/nanochemgpt
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - mongo
      - redis

  mongo:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=secure_password

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - nanochemgpt

volumes:
  mongo_data:
  redis_data:
```

**Create Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads logs retriever/index_doc retriever/index_passage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", "main_asgi:app"]
```

**Start with Docker Compose:**
```bash
# Set environment variables
export OPENAI_API_KEY=your_api_key_here

# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f nanochemgpt

# Stop services
docker-compose down
```

### 2. Using Docker Only

```bash
# Build image
docker build -t nanochemgpt .

# Run container
docker run -d \
  --name nanochemgpt \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_api_key_here \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  nanochemgpt

# Check status
docker ps
docker logs nanochemgpt
```

## Environment Configuration

### Complete .env Reference

```bash
# === Core Configuration ===
FLASK_ENV=production
DEBUG=False
SECRET_KEY=your_secure_secret_key_here
PORT=8000

# === OpenAI Configuration ===
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMB=text-embedding-3-small
OPENAI_MODEL=gpt-4

# === Embedding Configuration ===
EMBED_BACKEND=st  # Options: 'st' (sentence-transformers), 'openai'
EMBED_MODEL=all-MiniLM-L6-v2  # For sentence-transformers backend

# === Database Configuration ===
MONGODB_URI=mongodb://localhost:27017/nanochemgpt
REDIS_URL=redis://localhost:6379/0

# === File Storage ===
UPLOADS_DIR=data/uploads
MAX_FILE_SIZE=100MB
ALLOWED_EXTENSIONS=pdf,txt,docx,doc

# === Vector Store Configuration ===
RETRIEVER_INDEX_DIRS=retriever/index_doc,retriever/index_passage
RETRIEVER_INDEX_DIR_DOC=retriever/index_doc
RETRIEVER_LEVEL_DEFAULT=both
WEIGHT_DOC=0.6
WEIGHT_PASSAGE=0.4

# === NLP Models ===
SPACY_MODEL=harvester/miner/ner_model/model-best

# === Literature Mining ===
HARVESTER_MAX_RESULTS=200
HARVESTER_MIN_YEAR=2015
ENABLE_ENHANCED_RELEVANCE=true

# === Citation Management ===
ENABLE_ENHANCED_CITATIONS=true
CITATION_MIN_SCORE=0.25

# === Evaluation ===
JUDGE_MIN_HITS=1
JUDGE_MIN_SCORE=0.15
JUDGE_MIN_CHARS=64

# === Performance ===
WORKERS=4
WORKER_CONNECTIONS=1000
MAX_REQUESTS=1000
TIMEOUT=30

# === Logging ===
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOG_MAX_BYTES=10485760  # 10MB
LOG_BACKUP_COUNT=5

# === Security ===
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
CORS_ORIGINS=*

# === Development ===
PROFILING=False
MEMORY_PROFILING=False
```

## Data Setup

### 1. Initial Knowledge Base

```bash
# Download sample knowledge base (if available)
wget https://github.com/DMCarnahan/NanoChemGPT/releases/download/v1.0.0/sample_kb.tar.gz
tar -xzf sample_kb.tar.gz -C data/

# Or start with empty indexes
python retriever/index_jsonl.py --create-empty --output retriever/index_doc/
python retriever/index_jsonl.py --create-empty --output retriever/index_passage/
```

### 2. Literature Harvesting

```bash
# Initial literature harvest
python harvester/harvester.py \
  --config harvester/config.yaml \
  --max-results 100 \
  --output harvester/out_auto/initial_harvest.jsonl

# Index harvested literature
python retriever/index_jsonl.py \
  --input harvester/out_auto/initial_harvest.jsonl \
  --output retriever/index_doc/ \
  --update
```

### 3. Upload Test Documents

```bash
# Create test upload
mkdir -p data/test_uploads
echo "Sample synthesis protocol: Heat gold chloride solution to 100°C..." > data/test_uploads/sample.txt

# Test upload processing
python scripts/test_upload_simple.py
```

## Testing Installation

### 1. Basic Functionality Tests

```bash
# Test imports
python -c "
import app
import harvester.harvester
import retriever.retriever
import ai_eval.grader
print('All imports successful!')
"

# Test configuration
python -c "
from app import app
print(f'Flask app created: {app.name}')
print(f'Environment: {app.config.get(\"ENV\", \"unknown\")}')
"

# Test vector operations
python scripts/test_vector_store.py

# Test API endpoints
python scripts/test_vs_interface.py
```

### 2. Integration Tests

```bash
# Test question answering
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is nanochemistry?"}'

# Test health endpoint
curl "http://localhost:8000/health"

# Test upload functionality
curl -X POST "http://localhost:8000/upload" \
  -F "file=@data/test_uploads/sample.txt"
```

### 3. Performance Tests

```bash
# Load testing with Apache Bench
ab -n 100 -c 10 http://localhost:8000/health

# Memory usage monitoring
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Problem: ModuleNotFoundError
# Solution: Check virtual environment activation
source venv/bin/activate
pip install -r requirements.txt

# Problem: spaCy model not found
# Solution: Download model explicitly
python -m spacy download en_core_web_sm
```

**2. Memory Issues**
```bash
# Problem: Out of memory during vector operations
# Solution: Reduce batch sizes in .env
EMBED_BATCH_SIZE=32
RETRIEVER_BATCH_SIZE=16
```

**3. API Connection Issues**
```bash
# Problem: OpenAI API errors
# Solution: Check API key and credits
python -c "
import openai
openai.api_key = 'your_api_key'
try:
    openai.models.list()
    print('OpenAI connection successful')
except Exception as e:
    print(f'OpenAI error: {e}')
"
```

**4. File Permission Issues**
```bash
# Problem: Permission denied for data directories
# Solution: Fix permissions
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

**5. Port Conflicts**
```bash
# Problem: Port already in use
# Solution: Find and kill process or use different port
lsof -i :5000
kill -9 <PID>
# Or change PORT in .env
```

### Log Analysis

```bash
# View application logs
tail -f logs/app.log

# Check system logs
sudo journalctl -u nanochemgpt -f

# Monitor resource usage
htop
iotop
```

### Debugging Mode

```bash
# Enable debug mode
export FLASK_ENV=development
export DEBUG=True

# Run with verbose logging
python app.py --log-level DEBUG

# Profile memory usage
python -c "
import tracemalloc
tracemalloc.start()
# Your code here
current, peak = tracemalloc.get_traced_memory()
print(f'Current memory usage: {current / 1024 / 1024:.1f} MB')
print(f'Peak memory usage: {peak / 1024 / 1024:.1f} MB')
"
```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/DMCarnahan/NanoChemGPT/issues)
2. Review logs for specific error messages
3. Create a minimal reproduction case
4. Include system information:
   - OS and version
   - Python version
   - Installed package versions
   - Error messages and stack traces

---

This installation guide should provide everything needed to set up NanoChemGPT in various environments. For additional support, please refer to the documentation or create an issue on GitHub.