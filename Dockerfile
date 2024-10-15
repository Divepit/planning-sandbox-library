FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich  

# Install necessary dependencies including libGL and libglib2.0
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

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

# Expose port for TensorBoard
EXPOSE 6006

# Run your training script
CMD ["python3", "/app/ImitationLearning/IL_training.py"]