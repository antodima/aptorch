import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_emb(sin_inp):
    """Gets a base embedding for one dimension with sin and cos intertwined.
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def llada_loss(inputs, logits_pred, mask):
    """https://arxiv.org/pdf/2502.09992
    """
    batch_size, seq_len, vocab_size = logits_pred.shape
    pred_flat = logits_pred.view(batch_size * seq_len, vocab_size)
    target_flat = inputs.view(batch_size * seq_len)
    loss = F.cross_entropy(pred_flat, target_flat, reduction='none')
    loss = loss.view(batch_size, seq_len)
    loss = loss * mask.float()
    loss = loss.mean()
    loss = loss / mask.sum()

    return loss


class PositionalEncoding(nn.Module):
    def __init__(self, channels, dtype_override=None):
        super(PositionalEncoding, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        inv_freq = 1.0 / \
            (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.channels = channels
        self.dtype_override = dtype_override

    def forward(self, tensor):
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device,
                             dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros(
            (x, self.channels),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class DLM(nn.Module):
    """https://arxiv.org/pdf/2502.09992
    """

    def __init__(
        self,
        num_tokens: int,
        emb_dim: int,
        ff_dim: int,
        pad_idx: int,
        mask_idx: int,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx

        self.emb_token = nn.Embedding(
            num_embeddings=self.num_tokens,
            embedding_dim=self.emb_dim,
            padding_idx=self.pad_idx,
        )
        self.emb_time = PositionalEncoding(self.emb_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=1,
            dropout=0.1,
        )
        self.ff = nn.Sequential(
            nn.Linear(self.emb_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, self.emb_dim),
            nn.Dropout(0.1),
        )
        self.norm1 = nn.LayerNorm(self.emb_dim)
        self.norm2 = nn.LayerNorm(self.emb_dim)
        self.dropout = nn.Dropout(0.1)
        self.logits = nn.Linear(self.emb_dim, self.num_tokens)

    def forward(self, x, mask_ratio: float):
        batch_size, seq_len = x.shape
        mask_probs = torch.rand(batch_size, seq_len)
        mask = mask_probs < mask_ratio
        mask = mask & (x != self.pad_idx)
        x = torch.where(mask, self.mask_idx, x)

        x = self.emb_token(x)
        x = self.emb_time(x)
        attn_output, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            attn_mask=None,
        )
        x = x + self.dropout(attn_output)
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)
        logits = self.logits(x)

        return logits, mask.int()
