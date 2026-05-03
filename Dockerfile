FROM python:3.11-slim

WORKDIR /app

# Install specific qdrant-client version first
RUN pip install --no-cache-dir qdrant-client==1.17.1

# Copy requirements first (better caching)
COPY requirements.txt .

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]