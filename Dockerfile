FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and data
COPY src/ src/
COPY data/ data/

# Copy frontend build
COPY frontend/dist/ frontend/dist/

# Expose port
EXPOSE 7860

# Start FastAPI with uvicorn (WebSocket + static files)
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
