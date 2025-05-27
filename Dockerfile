# Dockerfile
# Containerize the API and training scripts then wire up GitHub Actions for CI/CD

# syntax=docker/dockerfile:1
##### BASE: Builder (for training & preprocessing)
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build-time deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-base.txt
RUN pip install --no-cache-dir -r requirements-base.txt

# Copy source entire project into the container
COPY ..

# Run preprocessing & training (optional entrypoint)
# You can override at runtime: 'docker run api:latest python src/preprocessing.py && python -m src.train ...'
ENTRYPOINT ["bash"]




##### RUNTIME: Serve only
FROM python:3.9-slim

WORKDIR /app

# Copy only the necessary files from builder
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir uvicorn fastapi joblib tensorflow scipy scikit-learn

# Copy processed artifacts, vectorizer & model
COPY --from=builder /app/data/processed ./data/processed
COPY --from=builder /app/models/production ./models/production
COPY --from=builder /app/src/api.py ./src/api.py

# Expose port
EXPOSE 8000

# Launch Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]