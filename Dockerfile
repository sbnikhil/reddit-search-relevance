FROM python:3.10-slim

WORKDIR /app

# System deps for faiss-cpu and LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render injects PORT; fall back to 8080 for local Docker runs
ENV PORT=8080

EXPOSE ${PORT}

# GCP credentials: set GOOGLE_APPLICATION_CREDENTIALS_JSON env var in Render
# dashboard (paste the full service account JSON). This script writes it to disk
# so google-auth can find it via GOOGLE_APPLICATION_CREDENTIALS.
CMD ["sh", "-c", "\
  if [ -n \"$GOOGLE_APPLICATION_CREDENTIALS_JSON\" ]; then \
    echo \"$GOOGLE_APPLICATION_CREDENTIALS_JSON\" > /tmp/gcp_creds.json && \
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp_creds.json; \
  fi && \
  exec uvicorn serving.app:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 30 \
"]
