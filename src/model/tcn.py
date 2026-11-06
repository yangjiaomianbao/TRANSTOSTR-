import torch.nn as nn
import torch.nn.functional as F


class CausalBlock(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, channels, kernel_size, dilation=dilation)
        self.skip = nn.Identity() if in_channels == channels else nn.Conv1d(in_channels, channels, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.skip(x) + self.dropout(self.activation(self.conv(F.pad(x, (self.left_padding, 0)))))


class SharedTCN(nn.Module):
    def __init__(self, d_model=64, kernel_size=3, dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        self.blocks = nn.Sequential(*[CausalBlock(2 if i == 0 else d_model, d_model,
            kernel_size, dilation, dropout) for i, dilation in enumerate(dilations)])

    def forward(self, x):
        b, n, t, c = x.shape
        h = self.blocks(x.reshape(b * n, t, c).transpose(1, 2))
        return h.transpose(1, 2).reshape(b, n, t, -1)
