import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.data import get_mask_from_lengths

def length_regulator(batch_seq, batch_dur, tgt_mask=None):
    # all tokens at masked idxes are taken care of with dur=0
    expanded = []
    out_lens = []
    max_len = batch_dur.sum(dim=1).max()
    if tgt_mask is not None:
        assert tgt_mask.shape[1] == max_len
    for seq, dur in zip(batch_seq, batch_dur):
        seq = seq.repeat_interleave(dur, dim=0)
        out_lens.append(seq.shape[0])
        expanded.append(
            F.pad(seq, (0,0,0, max_len - seq.shape[0]), "constant", 0.0)
        )

    expanded_batch = torch.stack(expanded)
    if tgt_mask is None:
        tgt_mask = get_mask_from_lengths(out_lens, max_len, device=expanded_batch.device)

    return expanded_batch, tgt_mask

class DurationPredictor(nn.Module):
    def __init__(self, d_model, n_filter, kernel_size, dropout_p):
        super().__init__()
        self.layers = nn.Sequential(
                Conv(d_model, n_filter, kernel_size, padding=(kernel_size - 1)//2 ),
                nn.ReLU(),
                nn.LayerNorm(n_filter),
                nn.Dropout(dropout_p),
                Conv(n_filter, n_filter, kernel_size, padding=1),
                nn.ReLU(),
                nn.LayerNorm(n_filter),
                nn.Dropout(dropout_p)
        )
        self.proj = nn.Linear(n_filter, 1)

    def forward(self, x, mask=None):
        out = self.layers(x)
        out = self.proj(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out
        
    
class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x