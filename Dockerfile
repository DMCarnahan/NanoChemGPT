### Builder: build a wheel for the package
FROM python:3.11-slim AS builder
WORKDIR /src

# Install system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install build tools
RUN python -m pip install --upgrade pip build wheel

# Copy pyproject/setup and requirements for caching
COPY pyproject.toml setup.cfg requirements.txt requirements-dev.txt ./

# Copy the rest of the source and build a wheel
COPY . .
RUN python -m build --wheel --no-isolation -o /dist


### Final runtime image
FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home appuser || true
WORKDIR /home/appuser/app

# Copy built wheel from builder
COPY --from=builder /dist /dist

# Install runtime deps and the built wheel
COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir /dist/*.whl

# Copy only application files (keep image small)
COPY . .

# Expose port and run as non-root
EXPOSE 8000
USER appuser

# Default command — keep simple and robust; platform providers can override CMD
CMD ["python", "app.py"]
