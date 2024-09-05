# Use the official PyTorch base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /home/workspace

# Install necessary system dependencies
RUN apt update && apt install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory (Tacotron2 code) into the container
COPY . /home/workspace

# Install required Python packages
RUN pip install deep_phonemizer matplotlib

# Create necessary directories
RUN mkdir data img

# Define the default command to run the training script
CMD ["python", "train.py"]
