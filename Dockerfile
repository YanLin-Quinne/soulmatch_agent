FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and data
COPY src/ src/
COPY data/ data/
COPY app.py .

# Expose port
EXPOSE 7860

# Start Gradio app (HuggingFace Spaces uses port 7860)
CMD ["python", "app.py"]
