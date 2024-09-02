docker run -it -v C:\Users\ryujm\Documents\workspace\Tacotron2:/home/workspace --name tacotron2 --gpus all pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
docker run -it -v /Users/moon/Documents/Tacotron2:/home/workspace --name tacotron2 pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

apt update
apt install build-essential -y
apt install git -y

pip install deep_phonemizer
pip install matplotlib