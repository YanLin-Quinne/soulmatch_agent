FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and data
COPY src/ src/
COPY data/ data/
COPY app.py .

# Copy Flask templates and frontend build
COPY templates/ templates/
COPY frontend/dist/ frontend/dist/

# Expose port
EXPOSE 7860

# Start Flask app (serves React frontend)
CMD ["python", "app.py"]
