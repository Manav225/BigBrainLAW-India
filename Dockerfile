FROM python:3.10-slim

# Install Git LFS
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Set working directory
WORKDIR /app

# Copy repo
COPY . /app

# Pull LFS files
RUN git lfs pull

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
