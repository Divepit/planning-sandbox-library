FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only requirements.txt to leverage Docker's cache
COPY requirements.txt /app/
COPY setup.py /app/
COPY README.md /app/

# Install Python dependencies (this layer will be cached if requirements.txt hasn't changed)
RUN pip3 install -e .

# Now copy the rest of the application code
COPY . /app

# Run your training script
CMD ["python3", "/app/NeuralNetwork/dataset_generation.py"]