from copy import deepcopy
from typing import Callable

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    """Based on LLaDa (https://arxiv.org/pdf/2502.09992).
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

    def forward(self, x):
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

        return logits

    @torch.no_grad()
    def sample(
        self,
        x,
        max_seq_len: int = 10,
        sampling_steps: int = 100,
    ):
        prompt_len = x.shape[1]
        if max_seq_len <= prompt_len:
            raise ValueError(
                "max_seq_len must be greater than prompt_len for generation.")

        if x.shape[0] != 1:
            raise ValueError(
                "Sampling method currently supports batch_size = 1.")

        initial_response_len = max_seq_len - prompt_len
        masked_response_part = torch.full(
            (1, initial_response_len), self.mask_idx, dtype=torch.long)

        current_sequence = torch.cat((x, masked_response_part), dim=-1)
        response_indices_slice = slice(prompt_len, max_seq_len)
        for step_idx in range(sampling_steps):
            next_t_val = 1.0 - ((step_idx + 1) / sampling_steps)
            logits = self.forward(current_sequence)
            predicted_tokens_all = torch.argmax(logits, dim=-1)
            masked_in_response = (
                current_sequence[:, response_indices_slice] == self.mask_idx)
            r0_candidate = current_sequence.clone()
            r0_candidate[:, response_indices_slice] = torch.where(
                masked_in_response,
                predicted_tokens_all[:, response_indices_slice],
                current_sequence[:, response_indices_slice]
            )
            num_tokens_to_be_masked_in_next_step = int(
                initial_response_len * next_t_val)
            num_tokens_to_be_masked_in_next_step = max(
                0, num_tokens_to_be_masked_in_next_step)
            next_sequence_step = r0_candidate.clone()
            response_logits = logits[:, response_indices_slice, :].squeeze(0)
            response_probs = F.softmax(response_logits, dim=-1)
            predicted_tokens_response = predicted_tokens_all[:, response_indices_slice].squeeze(
                0)
            # low confidence remasking strategy
            predicted_confidence = response_probs.gather(
                1, predicted_tokens_response.unsqueeze(-1)).squeeze(-1)
            sorted_confidences, sorted_indices_in_response = torch.sort(
                predicted_confidence, descending=False)
            relative_indices_to_remask = sorted_indices_in_response[
                :num_tokens_to_be_masked_in_next_step]
            full_indices_to_remask = response_indices_slice.start + relative_indices_to_remask
            next_sequence_step[:, full_indices_to_remask] = self.mask_idx
            current_sequence = next_sequence_step

            if (current_sequence[:, response_indices_slice] != self.mask_idx).all():
                break

        return current_sequence


class DLM_Pretrained(L.LightningModule):
    """Train the model on the masked prompt.
    """

    def __init__(
        self,
        num_tokens: int,
        emb_dim: int,
        ff_dim: int,
        pad_idx: int,
        mask_idx: int,
        mask_ratio: float,
        lr: float,
    ):
        super().__init__()
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.model = DLM(
            num_tokens=num_tokens,
            emb_dim=emb_dim,
            ff_dim=ff_dim,
            pad_idx=pad_idx,
            mask_idx=mask_idx,
        )

    def forward(self, inputs, target=None):
        return self.model(inputs)

    def _mask_inputs(self, inputs):
        batch_size, seq_len = inputs.shape
        mask_probs = torch.rand(batch_size, seq_len)
        mask = mask_probs < self.mask_ratio
        mask = mask & (inputs != self.pad_idx)
        mask_inputs = torch.where(mask, self.mask_idx, inputs)

        return mask_inputs, mask

    def loop_step(self, batch):
        inputs = batch
        mask_inputs, mask = self._mask_inputs(inputs)

        logits = self(mask_inputs)
        mask = mask.float()
        loss = torch.tensor(0.0)
        if mask.sum() != 0:
            loss = llada_loss(inputs, logits, mask) / self.mask_ratio

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loop_step(batch)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loop_step(batch)
        self.log("val_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def sample(self, inputs, max_seq_len, sampling_steps):
        prompt_len = inputs.shape[1]
        if max_seq_len <= prompt_len:
            raise ValueError(
                "max_seq_len must be greater than prompt_len for generation.")

        if inputs.shape[0] != 1:
            raise ValueError(
                "Sampling method currently supports batch_size = 1.")

        initial_response_len = max_seq_len - prompt_len
        masked_response_part = torch.full(
            (1, initial_response_len), self.mask_idx, dtype=torch.long)

        current_sequence = torch.cat((inputs, masked_response_part), dim=-1)
        response_indices_slice = slice(prompt_len, max_seq_len)
        for step_idx in range(sampling_steps):
            next_t_val = 1.0 - ((step_idx + 1) / sampling_steps)
            logits = self(current_sequence)
            predicted_tokens_all = torch.argmax(logits, dim=-1)
            masked_in_response = (
                current_sequence[:, response_indices_slice] == self.mask_idx)
            r0_candidate = current_sequence.clone()
            r0_candidate[:, response_indices_slice] = torch.where(
                masked_in_response,
                predicted_tokens_all[:, response_indices_slice],
                current_sequence[:, response_indices_slice]
            )
            num_tokens_to_be_masked_in_next_step = int(
                initial_response_len * next_t_val)
            num_tokens_to_be_masked_in_next_step = max(
                0, num_tokens_to_be_masked_in_next_step)
            next_sequence_step = r0_candidate.clone()
            response_logits = logits[:, response_indices_slice, :].squeeze(0)
            response_probs = F.softmax(response_logits, dim=-1)
            predicted_tokens_response = predicted_tokens_all[:, response_indices_slice].squeeze(
                0)
            # low confidence remasking strategy
            predicted_confidence = response_probs.gather(
                1, predicted_tokens_response.unsqueeze(-1)).squeeze(-1)
            sorted_confidences, sorted_indices_in_response = torch.sort(
                predicted_confidence, descending=False)
            relative_indices_to_remask = sorted_indices_in_response[
                :num_tokens_to_be_masked_in_next_step]
            full_indices_to_remask = response_indices_slice.start + relative_indices_to_remask
            next_sequence_step[:, full_indices_to_remask] = self.mask_idx
            current_sequence = next_sequence_step

            if (current_sequence[:, response_indices_slice] != self.mask_idx).all():
                break

        return current_sequence
