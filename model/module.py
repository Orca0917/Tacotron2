import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Tuple, Optional, List
from .utils import _get_conv1d_layer, _get_linear_layer, _get_mask_from_lengths

class _Encoder(nn.Module):

    """
    인코더는 문자 시퀀스를 숨겨진 특징 표현으로 변환한다.
    """
    
    def __init__(
        self,
        encoder_embedding_dim: int,
        encoder_n_convolution: int,
        encoder_kernel_size: int,
    ) -> None:
        super().__init__()

        self.convolutions = nn.ModuleList()
        for _ in range(encoder_n_convolution):
            conv_layer = nn.Sequential(
                _get_conv1d_layer(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),

                nn.BatchNorm1d(encoder_embedding_dim),
            )
            self.convolutions.append(conv_layer)

        self.lstm = nn.LSTM(
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

        # 파라미터들이 메모리 공간에 인접하여 존재할 수 있도록 설정 (최적화)
        self.lstm.flatten_parameters()

    def forward(self, x: Tensor, input_lengths: Tensor) -> Tensor:

        # 각 필터가 5개의 문자를 아우르는 형태의 512개의 필터를 포함한 3개의 컨볼루션 층을 거친다.
        # 각 층들 이후에는 배치 정규화와 ReLU 활성함수가 적용된다.
        for conv in self.convolutions: # [B, 256, seq_len]
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)

        # 최종 컨볼루션 층의 출력은 512개의 유닛(각 방향에 256개)을 가진 단일 양방향 LSTM 층으로 전달되어 인코딩된 특징을 생성한다.
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs
    

class _Decoder(nn.Module):

    def __init__(
        self,
        n_mels: int,
        n_frames_per_step: int,
        encoder_embedding_dim: int,
        decoder_rnn_dim: int,
        decoder_max_step: int,
        decoder_dropout: float,
        decoder_early_stopping: bool,
        attention_rnn_dim: int,
        attention_hidden_dim: int,
        attention_location_n_filter: int,
        attention_location_kernel_size: int,
        attention_dropout: float,
        prenet_dim: int,
        gate_threshold: float,
    ) -> None:
        
        super().__init__()
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.decoder_max_step = decoder_max_step
        self.gate_threshold = gate_threshold
        self.attention_dropout = attention_dropout
        self.decoder_dropout = decoder_dropout
        self.decoder_early_stopping = decoder_early_stopping

        self.prenet = _Prenet(n_mels * n_frames_per_step, [prenet_dim, prenet_dim])

        self.attention_rnn = nn.LSTMCell(prenet_dim + encoder_embedding_dim, attention_rnn_dim)

        self.attention_layer = _Attention(
            attention_rnn_dim,
            encoder_embedding_dim,
            attention_hidden_dim,
            attention_location_n_filter,
            attention_location_kernel_size
        )

        self.decoder_rnn = nn.LSTMCell(attention_rnn_dim + encoder_embedding_dim, decoder_rnn_dim, True)

        self.linear_projection = _get_linear_layer(decoder_rnn_dim + encoder_embedding_dim, n_mels * n_frames_per_step)

        self.gate_layer = _get_linear_layer(
            decoder_rnn_dim + encoder_embedding_dim, 1, bias=True, w_init_gain="sigmoid"
        )

    def forward(
        self, memory: Tensor, mel_specgram_truth: Tensor, memory_lengths: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        decoder_input = self._get_initial_frame(memory).unsqueeze(0)
        decoder_inputs = self._parse_decoder_inputs(mel_specgram_truth)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask = _get_mask_from_lengths(memory_lengths)
        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self._initialize_decoder_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_specgram, gate_outputs, alignments = self._parse_decoder_outputs(
            torch.stack(mel_outputs), torch.stack(gate_outputs), torch.stack(alignments)
        )

        return mel_specgram, gate_outputs, alignments

    def _get_initial_frame(self, memory: Tensor) -> Tensor:

        n_batch = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(n_batch, self.n_mels * self.n_frames_per_step, dtype=dtype, device=device)
        return decoder_input
    
    def _parse_decoder_inputs(self, decoder_inputs: Tensor) -> Tensor:

        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1
        )

        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs
    
    def _parse_decoder_outputs(
            self, mel_specgram: Tensor, gate_outputs: Tensor, alignments: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        alignments = alignments.transpose(0, 1).contiguous()
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        mel_specgram = mel_specgram.transpose(0, 1).contiguous()
        shape = (mel_specgram.shape[0], -1, self.n_mels)
        mel_specgram = mel_specgram.view(*shape)
        mel_specgram = mel_specgram.transpose(1, 2)

        return mel_specgram, gate_outputs, alignments

    def _initialize_decoder_states(
        self, memory: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        n_batch = memory.size(0)
        max_time = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(n_batch, self.attention_rnn_dim, dtype=dtype, device=device)
        attention_cell = torch.zeros(n_batch, self.attention_rnn_dim, dtype=dtype, device=device)

        decoder_hidden = torch.zeros(n_batch, self.decoder_rnn_dim, dtype=dtype, device=device)
        decoder_cell = torch.zeros(n_batch, self.decoder_rnn_dim, dtype=dtype, device=device)

        attention_weights = torch.zeros(n_batch, max_time, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(n_batch, max_time, dtype=dtype, device=device)
        attention_context = torch.zeros(n_batch, self.encoder_embedding_dim, dtype=dtype, device=device)

        processed_memory = self.attention_layer.memory_layer(memory)

        return (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory
        )
    
    def decode(
        self,
        decoder_input: Tensor,
        attention_hidden: Tensor,
        attention_cell: Tensor,
        decoder_hidden: Tensor,
        decoder_cell: Tensor,
        attention_weights: Tensor,
        attention_weights_cum: Tensor,
        attention_context: Tensor,
        memory: Tensor,
        processed_memory: Tensor,
        mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        cell_input = torch.cat((decoder_input, attention_context), -1)
        attention_hidden, attention_cell = self.attention_rnn(cell_input, (attention_hidden, attention_cell))
        attention_hidden = F.dropout(attention_hidden, self.attention_dropout, self.training)

        attention_weights_cat = torch.cat((attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)), dim=1)
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, memory, processed_memory, attention_weights_cat, mask
        )

        attention_weights_cum += attention_weights
        decoder_input = torch.cat((attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(decoder_input, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(decoder_hidden, self.decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((decoder_hidden, attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (
            decoder_output,
            gate_prediction,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
        )


class _Prenet(nn.Module):
    
    def __init__(self, in_dim: int, out_sizes: List[int]) -> None:
        super().__init__()
        in_sizes = [in_dim] + out_sizes[:-1]
        self.layers = nn.ModuleList(
            [_get_linear_layer(in_size, out_size, bias=False) for (in_size, out_size) in zip(in_sizes, out_sizes)]
        )

    def forward(self, x: Tensor) -> Tensor:
        
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class _Attention(nn.Module):

    def __init__(
        self,
        attention_rnn_dim: int,
        encoder_embedding_dim: int,
        attention_hidden_dim: int,
        attention_location_n_filter: int,
        attention_location_kernel_size: int,
    ) -> None:
        super().__init__()
        self.query_layer = _get_linear_layer(
            attention_rnn_dim, attention_hidden_dim, bias=False, w_init_gain="tanh"
        )
        self.memory_layer = _get_linear_layer(
            encoder_embedding_dim, attention_hidden_dim, bias=False, w_init_gain="tanh"
        )
        self.v = _get_linear_layer(attention_hidden_dim, 1, bias=False)
        self.location_layer = _LocationLayer(
            attention_location_n_filter,
            attention_location_kernel_size,
            attention_hidden_dim
        )
        self.score_mask_value = -float("inf")

    def _get_alignment_energies(self, query: Tensor, processed_memory: Tensor, attention_weights_cat: Tensor) -> Tensor:
        
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attentioni_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attentioni_weights + processed_memory))

        alignment = energies.squeeze(2)
        return alignment
    
    def forward(
        self,
        attention_hidden_state: Tensor,
        memory: Tensor,
        processed_memory: Tensor,
        attention_weights_cat: Tensor,
        mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        alignment = self._get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)
        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
    

class _LocationLayer(nn.Module):

    def __init__(
        self,
        attention_n_filter: int,
        attention_kernel_size: int,
        attention_hidden_dim: int,
    ):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = _get_conv1d_layer(
            2,
            attention_n_filter,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = _get_linear_layer(
            attention_n_filter, attention_hidden_dim, bias=False, w_init_gain="tanh"
        )
    
    def forward(self, attention_weights_cat: Tensor) -> Tensor:
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention
    

class _Postnet(nn.Module):

    def __init__(
        self,
        n_mels: int,
        postnet_embedding_dim: int,
        postnet_kernel_size: int,
        postnet_n_convolution: int,
    ):
        super().__init__()
        self.convolutions = nn.ModuleList()

        for i in range(postnet_n_convolution):
            in_channels = n_mels if i == 0 else postnet_embedding_dim
            out_channels = n_mels if i == (postnet_n_convolution - 1) else postnet_embedding_dim
            init_gain = "linear" if i == (postnet_n_convolution - 1) else "tanh"
            num_features = n_mels if i == (postnet_n_convolution - 1) else postnet_embedding_dim

            self.convolutions.append(
                nn.Sequential(
                    _get_conv1d_layer(
                        in_channels,
                        out_channels,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain=init_gain,
                    ),
                    nn.BatchNorm1d(num_features)
                )
            )

        self.n_convs = len(self.convolutions)


    def forward(self, x: Tensor) -> Tensor:

        for i, conv in enumerate(self.convolutions):
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)

        return x