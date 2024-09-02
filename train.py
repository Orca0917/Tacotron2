import torch
import torchaudio

from torch.utils.data import DataLoader
from torchaudio.datasets import LJSPEECH
from dataset import Tacotron2Dataset, Tacotron2Collate
from utils import plot_specgram

def main():
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 준비: LJSpeech 데이터셋 다운로드
    dataset = LJSPEECH(root="./data", download=True)


    # Tacotron2 bundle 준비 (Text 전처리기 & Vocoder)
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

    processor = bundle.get_text_processor()
    vocoder = bundle.get_vocoder().to(device)

    # Tacotron2 모델 학습용 데이터셋 준비
    tacotron2dataset = Tacotron2Dataset(dataset, processor)
    tacotron2dataloader = DataLoader(
        dataset=tacotron2dataset, 
        batch_size=4,
        shuffle=True,
        drop_last=True,
        collate_fn=Tacotron2Collate()
    )

    # 옵티마이저, 모델, 손실함수

    # 모델 학습
    for batch in tacotron2dataloader:

        (
            text,
            text_length,
            mel_specgram,
            mel_specgram_length,
            gate
        ) = batch

        for i, mel in enumerate(mel_specgram):
            plot_specgram(mel, file_path=f"{i}.png") 
        break


if __name__ == "__main__":
    main()