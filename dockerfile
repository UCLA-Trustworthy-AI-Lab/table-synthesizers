# Use Python runtime as the base image
FROM python:3.10-slim

# Set environment variables for paths
ENV SYNTHESIZER_CONFIG_PATH=/app/config/synthesizer_config.json
ENV TRANSFORMED_DATA_PATH=/app/data/transformed
ENV SYNTHESIZER_OUTPUT_PATH=/app/output

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define volumes for data input and output
VOLUME /app/data
VOLUME /app/output

# Expose a port if needed (optional, not used in this case)
# EXPOSE 8080

# Define the entry point for the container
ENTRYPOINT ["python", "main.py"]
