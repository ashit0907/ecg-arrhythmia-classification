
FROM python:3.12-slim
WORKDIR /app

# Install CPU-only PyTorch separately
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY app/ ./app/
COPY models/lstm_plain.pt ./models/lstm_plain.pt

# Cloud Run listens on port 8080
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
