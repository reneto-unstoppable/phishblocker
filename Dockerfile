# Base Python image
FROM python:3.10-slim

# Set work directory inside container
WORKDIR /app

# Install system dependencies for gdown and others
RUN apt-get update && \
    apt-get install -y wget curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Run your app
CMD ["python", "app.py"]
