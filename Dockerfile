FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich  
# Set a default timezone to avoid tzdata prompts

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

WORKDIR /app

COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port for TensorBoard
EXPOSE 6006

# Run your training script
CMD ["python3", "/app/ImitationLearning/IL_training.py"]

