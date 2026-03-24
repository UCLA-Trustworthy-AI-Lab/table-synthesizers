# Use Python runtime as the base image
FROM python:3.10-slim

# Set environment variables for paths
ENV SYNTHESIZER_CONFIG_PATH=/app/config/synthesizer_config.json
ENV TRANSFORMED_DATA_PATH=/app/data/transformed
ENV SYNTHESIZER_OUTPUT_PATH=/app/output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
# Combine install and cleanup to reduce layer size
RUN pip install --no-cache-dir -r requirements.txt

# Install synthcity from custom branch for torch compatibility
RUN git clone -b main https://github.com/ohsono/synthcity.git /tmp/synthcity && \
  pip install --no-cache-dir -e /tmp/synthcity && \
  rm -rf /tmp/synthcity

# Copy the rest of the application
COPY . .

# Define volumes for data input and output
VOLUME /app/data
VOLUME /app/output

# Entry point
ENTRYPOINT ["python", "main.py"]
