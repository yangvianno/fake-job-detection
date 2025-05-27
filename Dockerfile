# Dockerfile
# syntax=docker/dockerfile:1

FROM python:3.9-slim

# Defines which requirements file to install (defaults to requirements-base.txt)
ARG REQUIREMENTS=requirements-base.txt      

WORKDIR /app

# Ensures Docker can resolve the build-arg at build time and grabs exactly that file
COPY ${REQUIREMENTS} ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS}

# Brings in everything else (your vectorizers, models, src/, etc)
COPY . .

# 4) expose and launch Uvicorn
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]