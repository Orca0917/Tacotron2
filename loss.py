from torch import nn, Tensor
from typing import Tuple

class Tacotron2Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self, 
        output: Tuple[Tensor, Tensor, Tensor, Tensor], 
        target: Tuple[Tensor, Tensor]
    ):

        mel_target, gate_target = target
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_specgram, mel_specgram_postnet, gate_outputs, _ = output
        gate_outputs = gate_outputs.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_specgram, mel_target) + \
            nn.MSELoss()(mel_specgram_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_outputs, gate_target)
        return mel_loss + gate_loss