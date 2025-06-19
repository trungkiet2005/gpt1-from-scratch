FROM python:3.9-slim

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

# Copy application files
COPY app.py .
COPY model.py .

# Copy dataset if it exists, otherwise create a placeholder
COPY dataset.txt* ./

# Copy trained model and encoded data if they exist
COPY encoded_data.pt* ./
COPY final_model.pt* ./
COPY model_checkpoint.pt* ./
COPY best_model.pt* ./

# Create a placeholder dataset if none exists
RUN if [ ! -f "dataset.txt" ]; then \
    echo "xin chào thế giới. đây là một bài thơ đơn giản. tình yêu và cuộc sống luôn tươi đẹp." > dataset.txt; \
    fi

# Check for model files and create info
RUN echo "🔍 Checking for model files..." && \
    ls -la *.pt 2>/dev/null || echo "⚠️  No trained model files found. Will use randomly initialized model." && \
    if [ -f "final_model.pt" ]; then echo "✅ Found final_model.pt"; fi && \
    if [ -f "encoded_data.pt" ]; then echo "✅ Found encoded_data.pt"; fi

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Command to run the application
CMD ["python", "-c", "import uvicorn; uvicorn.run('app:app', host='0.0.0.0', port=7860)"] 