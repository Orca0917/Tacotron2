import torch

from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Tuple
from torchaudio import transforms
from torchaudio.pipelines._tts.interface import Tacotron2TTSBundle


def phonemizer(
        text: str = None,
        processor: Tacotron2TTSBundle.TextProcessor = None
    ) -> Tuple[Tensor, Tensor]:

    """
    Grapheme 텍스트를 전처리기를 사용하여 phoneme 텍스트로 변환해주는 함수    
    """

    processed, lengths = processor(text)
    return processed, lengths


def wav2mel(
        audio: Tensor = None,
        sample_rate: int = 22050,
) -> Tensor:
    """
    Waveform 형식의 음성을 멜스펙트로그램으로 변환해주는 함수
    """

    frame_size = int(0.05 * sample_rate)
    frame_hop = int(0.0125 * sample_rate)

    mel_specgram_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=80,
        n_fft=frame_size,
        hop_length=frame_hop,
        f_min=125,
        f_max=7600,
        window_fn=torch.hann_window
    )

    mel_specgram = mel_specgram_transform(audio)
    mel_specgram = torch.clamp(mel_specgram, min=0.01)
    mel_specgram = torch.log(mel_specgram)

    return mel_specgram, torch.LongTensor([mel_specgram.size(-1)])
        

class Tacotron2Dataset(Dataset):

    def __init__(
            self,
            dataset: Dataset = None,
            text_processor: Tacotron2TTSBundle.TextProcessor = None
        ) -> None:

        self.dataset = dataset
        self.processor = text_processor


    def __getitem__(self, index):

        (
            processed_text, 
            processed_text_length
        ) = phonemizer(self.dataset[index][3], self.processor)

        (
            processed_mel_specgram,
            processed_mel_specgram_length
        ) = wav2mel(self.dataset[index][0].squeeze(), self.dataset[index][1])


        return (
            processed_text.squeeze(),
            processed_text_length,
            processed_mel_specgram,
            processed_mel_specgram_length
        )

    def __len__(self):
        return len(self.dataset)
    

class Tacotron2Collate():
    
    def __init__(self):
        ...

    def __call__(self, batch: List[Tensor]):

        B = len(batch)

        # 텍스트의 가장 긴 길이를 찾기
        text_lengths = torch.LongTensor([text_len for _, text_len, _, _ in batch])
        text_lengths_sorted, ids_sorted_decreasing = torch.sort(text_lengths, descending=True, dim=0)
        max_length = text_lengths_sorted[0]

        text_padded = torch.LongTensor(B, max_length)
        text_padded.zero_()

        # zero padded 된 텍스트 시퀀스 배치 구성하기
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # 멜 스펙트로그램의 가장 긴 길이를 찾기
        n_mels = batch[0][2].size(0)
        max_melspec_len = max([melspec.size(1) for _, _, melspec, _ in batch])
        
        mel_padded = torch.Tensor(B, n_mels, max_melspec_len)
        mel_padded.zero_()
        gate_padded = torch.Tensor(B, max_melspec_len)
        gate_padded.zero_()
        mel_lengths = torch.LongTensor(B)

        # zero padded 된 멜 스펙트로그램 만들기 (+ 종료 토큰과, 멜 스펙트로그램 시퀀스 길이도 반환)
        for i in range(len(ids_sorted_decreasing)):
            melspec = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, :, :melspec.size(1)] = melspec
            gate_padded[i, melspec.size(1) - 1:] = 1
            mel_lengths[i] = melspec.size(1)
        
        return (
            text_padded,
            text_lengths,
            mel_padded,
            mel_lengths,
            gate_padded
        )

