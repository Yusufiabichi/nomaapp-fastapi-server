# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency definitions first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Run the web service on container startup using the dynamic port assigned by Cloud Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]