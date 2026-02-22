FROM python:3.11-slim

WORKDIR /app

# Install Node.js 18.x for frontend build
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install (without heavy ML deps)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend and build
COPY frontend/package*.json frontend/
WORKDIR /app/frontend
RUN npm install
COPY frontend/ .
RUN npm run build

# Copy backend code
WORKDIR /app
COPY src/ src/
COPY data/ data/

# Expose port
EXPOSE 7860

# Start backend (HuggingFace Spaces uses port 7860)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--ws-ping-interval", "20", "--ws-ping-timeout", "30"]
