import torch
import torchaudio

from torch.utils.data import DataLoader
from torchaudio.datasets import LJSPEECH
from dataset import Tacotron2Dataset, Tacotron2Collate
from utils import plot_specgram
from model.tacotron2 import Tacotron2
from loss import Tacotron2Loss
from torch.utils.data import random_split

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

    # 데이터를 train과 valid로 80:20 비율로 나누기
    train_size = int(0.8 * len(tacotron2dataset))
    valid_size = len(tacotron2dataset) - train_size

    # random_split을 사용해 train과 valid 데이터셋 나누기
    train_dataset, valid_dataset = random_split(tacotron2dataset, [train_size, valid_size])

    tacotron2_train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=8,
        shuffle=True,
        drop_last=True,
        collate_fn=Tacotron2Collate()
    )

    tacotron2_test_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        collate_fn=Tacotron2Collate()
    )

    # 옵티마이저, 모델, 손실함수
    model = Tacotron2().to(device)
    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    epochs = 20
    global_step = 0

    # 모델 학습
    for epoch in range(epochs):

        model.train()
        for step, batch in enumerate(tacotron2_train_dataloader):

            (
                text,
                text_length,
                mel_specgram,
                mel_specgram_length,
                gate
            ) = batch

            text = text.to(device)
            text_length = text_length.to(device)
            mel_specgram = mel_specgram.to(device)
            mel_specgram_length = mel_specgram_length.to(device)
            gate = gate.to(device)

            pred = model(text, text_length, mel_specgram, mel_specgram_length)

            loss = criterion(pred, (mel_specgram, gate))

            optimizer.zero_grad()
            loss.backward()

            # 그래디언트 클리핑
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            print(f"{epoch + 1:<5} | {global_step:<10}({(step * 100) / len(tacotron2_train_dataloader): .1f}%) | {loss.item():<12.6f} | {grad_norm.item():<12.6f}")
            global_step += 1

            if step % 20 == 0:
                for i, mel in enumerate(pred[0]):
                    plot_specgram(mel, file_path=f"img/step{global_step}-{i}.png")
                    break

                for i, mel in enumerate(pred[0]):
                    plot_specgram(mel, file_path=f"img/postnet-step{global_step}-{i}.png")
                    break

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tacotron2_test_dataloader):

                (
                    text,
                    text_length,
                    mel_specgram,
                    mel_specgram_length,
                    gate
                ) = batch

                text = text.to(device)
                text_length = text_length.to(device)
                mel_specgram = mel_specgram.to(device)
                mel_specgram_length = mel_specgram_length.to(device)
                gate = gate.to(device)

                pred = model(text, text_length, mel_specgram, mel_specgram_length)

                loss = criterion(pred, (mel_specgram, gate))

                print(f"{epoch + 1:<5} | Validation loss {loss.item() :<10}")



if __name__ == "__main__":
    main()