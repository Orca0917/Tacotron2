from typing import Tuple, Optional
from .module import _Encoder, _Decoder, _Postnet
from .utils import _get_mask_from_lengths

import torch
from torch import nn, Tensor


class Tacotron2(nn.Module):

    def __init__(
        self,
        mask_padding: bool = False,
        n_mels: int = 80,
        n_symbol: int = 148,
        n_frames_per_step: int = 1,
        symbol_embedding_dim: int = 512,
        encoder_embedding_dim: int = 512,
        encoder_n_convolution: int = 3,
        encoder_kernel_size: int = 5,
        decoder_rnn_dim: int = 1024,
        decoder_max_step: int = 2000,
        decoder_dropout: float = 0.1,
        decoder_early_stopping: bool = True,
        attention_rnn_dim: int = 1024,
        attention_hidden_dim: int = 128,
        attention_location_n_filter: int = 32,
        attention_location_kernel_size: int = 31,
        attention_dropout: float = 0.1,
        prenet_dim: int = 256,
        postnet_n_convolution: int = 5,
        postnet_kernel_size: int = 5,
        postnet_embedding_dim: int = 512,
        gate_threshold: float = 0.5,
    ) -> None:
        super().__init__()

        self.mask_padding = mask_padding
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.embedding = nn.Embedding(n_symbol, symbol_embedding_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.encoder = _Encoder(encoder_embedding_dim, encoder_n_convolution, encoder_kernel_size)
        self.decoder = _Decoder(
            n_mels,
            n_frames_per_step,
            encoder_embedding_dim,
            decoder_rnn_dim,
            decoder_max_step,
            decoder_dropout,
            decoder_early_stopping,
            attention_rnn_dim,
            attention_hidden_dim,
            attention_location_n_filter,
            attention_location_kernel_size,
            attention_dropout,
            prenet_dim,
            gate_threshold
        )
        self.postnet = _Postnet(n_mels, postnet_embedding_dim, postnet_kernel_size, postnet_n_convolution)

    def forward(
        self,
        tokens: Tensor,
        token_lengths: Tensor,
        mel_specgram: Tensor,
        mel_specgram_lengths: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        # 입력 문자는 학습된 512차원 문자 임베딩을 사용하여 표현된다.
        embedded_inputs = self.embedding(tokens).transpose(1, 2) # [B, 256, seq_len]
        encoder_outputs = self.encoder(embedded_inputs, token_lengths)

        # 인코더의 출력은 디코더의 각 출력 단계에서 고정된 길이의 컨텍스트 벡터로
        # 전체 인코딩된 시퀀스를 요약하는 어텐션 네트워크에 의해 사용된다.

        mel_specgram, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_specgram, memory_lengths=token_lengths
        )

        mel_specgram_postnet = self.postnet(mel_specgram)
        mel_specgram_postnet = mel_specgram + mel_specgram_postnet

        if self.mask_padding:
            mask = _get_mask_from_lengths(mel_specgram_lengths)
            mask = mask.expand(self.n_mels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            mel_specgram.masked_fill_(mask, 0.0)
            mel_specgram_postnet.masked_fill_(mask, 0.0)
            gate_outputs.masked_fill_(mask[:, 0, :], 1e3)

        return mel_specgram, mel_specgram_postnet, gate_outputs, alignments
    
    @torch.jit.export
    def infer(self, tokens: Tensor, lengths: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        n_batch, max_length = tokens.shape
        if lengths is None:
            lengths = torch.tensor([max_length]).expand(n_batch).to(tokens.device, tokens.dtype)

        assert lengths is not None
        embedded_inputs = self.embedding(tokens).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, lengths)
        mel_specgram, mel_specgram_lengths, _, alignments = self.decoder.infer(encoder_outputs, lengths)

        mel_outputs_postnet = self.postnet(mel_specgram)
        mel_outputs_postnet = mel_specgram + mel_outputs_postnet

        alignments = alignments.unfold(1, n_batch, n_batch).transpose(0, 2)

        return mel_outputs_postnet, mel_specgram_lengths, alignments
