# Tacotron2 Implementation (Unofficial)

This repository provides an unofficial implementation of Tacotron2 using PyTorch. For a more detailed and accurate implementation, it is strongly recommended to refer to NVIDIA’s official Tacotron2 repository. This current implementation serves as an acoustic model to generate mel-spectrograms. To generate actual speech, you will need to integrate a vocoder model provided by torchaudio or retrain one. Most of the code in this repository is based on NVIDIA’s Tacotron2 implementation.

<br>

## 1. Getting Started

### Prerequisites
- **Docker** and **NVIDIA GPU Drivers** (for GPU support)

### Setup Instructions

1. Clone the repository:

    ```
    git clone https://github.com/Orca0917/Tacotron2.git
    cd Tacotron2
    ```

2. Build the Docker image:

    ```
    docker build -t tacotron2 .
    ```

3. Run the Docker container:

    ```
    docker run -it --name tacotron2-container --gpus all tacotron2
    ```

4. Train the model inside the container:

    ```
    python train.py
    ```


<br>

## 2. Result

Once the model is trained, you will obtain a mel-spectrogram as the output of the acoustic model, which can be visualized similarly to the image below:

![alt text](tacotron2-result.png)

<br>

## 3. Additional Notes
This implementation focuses on generating mel-spectrograms. To complete the text-to-speech pipeline, you will need to use a vocoder (e.g., WaveNet, Griffin-Lim, or a model from torchaudio) to convert the spectrograms into waveform audio.

<br>

## 4. References

[1] Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions. [[Link to Paper]](https://arxiv.org/pdf/1712.05884)

[2] github NVIDIA/Tacotron2 [[Link to GitHub]](https://github.com/NVIDIA/tacotron2)
