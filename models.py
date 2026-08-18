from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn


def set_cpu_mode() -> None:
    # Deliberately CPU-only for this experiment.
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def make_batch(
    batch_size: int,
    min_len: int,
    max_len: int,
    pad_to: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      tokens: [B, pad_to], values 0..3, padded with 4
      lengths: [B]
      targets: [B], exact composition sum mod 4
    """
    lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)
    tokens = torch.full((batch_size, pad_to), 4, dtype=torch.long, device=device)

    for i, L in enumerate(lengths.tolist()):
        tokens[i, :L] = torch.randint(0, 4, (L,), device=device)

    mask = tokens != 4
    safe = tokens.clamp(max=3)
    targets = (safe * mask.long()).sum(dim=1) % 4
    return tokens, lengths, targets


class StructuredPhaseModel(nn.Module):
    """
    A tiny trainable relational model.

    It does NOT know the four input relation phases initially.
    It learns one angle per relation symbol from short examples.

    Composition is structural:
        theta_path = sum(theta_edge)

    That is the inductive bias being tested.
    """

    def __init__(self):
        super().__init__()
        self.angles = nn.Parameter(torch.randn(4) * 0.35)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens != 4
        safe = tokens.clamp(max=3)
        edge_angles = self.angles[safe] * mask.float()
        theta = edge_angles.sum(dim=1)
        # Return a unit-vector representation of the resulting phase.
        return torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)

    def predict_class(self, tokens: torch.Tensor) -> torch.Tensor:
        pred_xy = self(tokens)
        canonical = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=pred_xy.dtype,
            device=pred_xy.device,
        )
        scores = pred_xy @ canonical.T
        return scores.argmax(dim=1)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TinyTransformer(nn.Module):
    """
    Generic learned baseline. It gets no explicit composition rule.
    It must infer the sequence operation from training data.

    Sinusoidal rather than learned positions are used so test positions beyond
    the training lengths are at least defined.
    """

    def __init__(
        self,
        max_len: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dim_feedforward: int = 128,
    ):
        super().__init__()
        # ids 0..3 = relation symbols, 4 = PAD, 5 = CLS
        self.embedding = nn.Embedding(6, d_model)
        self.pos = SinusoidalPositionEncoding(d_model, max_len + 1)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.size(0)
        cls = torch.full((B, 1), 5, dtype=torch.long, device=tokens.device)
        seq = torch.cat([cls, tokens], dim=1)

        pad_mask = seq == 4
        x = self.embedding(seq)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(x[:, 0, :])


def phase_targets(targets: torch.Tensor) -> torch.Tensor:
    canonical = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float32,
        device=targets.device,
    )
    return canonical[targets]


def train_models(
    *,
    seed: int = 7,
    train_min_len: int = 1,
    train_max_len: int = 5,
    test_max_len: int = 32,
    steps: int = 700,
    batch_size: int = 192,
    progress_callback=None,
) -> Tuple[StructuredPhaseModel, TinyTransformer, Dict[str, List[float]]]:
    set_cpu_mode()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    phase_model = StructuredPhaseModel().to(device)
    transformer = TinyTransformer(max_len=test_max_len).to(device)

    phase_opt = torch.optim.Adam(phase_model.parameters(), lr=0.03)
    trans_opt = torch.optim.AdamW(transformer.parameters(), lr=2e-3, weight_decay=1e-4)

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    history = {"phase_loss": [], "transformer_loss": []}

    for step in range(1, steps + 1):
        tokens, lengths, targets = make_batch(
            batch_size,
            train_min_len,
            train_max_len,
            test_max_len,
            device,
        )

        # Structured phase model
        phase_opt.zero_grad(set_to_none=True)
        pred_xy = phase_model(tokens)
        loss_phase = mse(pred_xy, phase_targets(targets))
        loss_phase.backward()
        phase_opt.step()

        # Generic transformer
        trans_opt.zero_grad(set_to_none=True)
        logits = transformer(tokens)
        loss_trans = ce(logits, targets)
        loss_trans.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        trans_opt.step()

        history["phase_loss"].append(float(loss_phase.detach().cpu()))
        history["transformer_loss"].append(float(loss_trans.detach().cpu()))

        if progress_callback and (step == 1 or step % 25 == 0 or step == steps):
            progress_callback(
                step,
                steps,
                float(loss_phase.detach().cpu()),
                float(loss_trans.detach().cpu()),
            )

    return phase_model, transformer, history


@torch.no_grad()
def evaluate_by_length(
    phase_model: StructuredPhaseModel,
    transformer: TinyTransformer,
    *,
    min_len: int = 1,
    max_len: int = 32,
    examples_per_length: int = 500,
    seed: int = 1234,
) -> Dict[str, List[float]]:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    phase_model.eval()
    transformer.eval()

    lengths_out = []
    exact_acc = []
    phase_acc = []
    transformer_acc = []

    for L in range(min_len, max_len + 1):
        tokens, lengths, targets = make_batch(
            examples_per_length, L, L, max_len, device
        )

        # Exact relational closure is by construction the target.
        exact = ((tokens.clamp(max=3) * (tokens != 4).long()).sum(dim=1) % 4)
        exact_accuracy = (exact == targets).float().mean().item()

        phase_pred = phase_model.predict_class(tokens)
        phase_accuracy = (phase_pred == targets).float().mean().item()

        logits = transformer(tokens)
        trans_pred = logits.argmax(dim=1)
        trans_accuracy = (trans_pred == targets).float().mean().item()

        lengths_out.append(L)
        exact_acc.append(exact_accuracy)
        phase_acc.append(phase_accuracy)
        transformer_acc.append(trans_accuracy)

    return {
        "length": lengths_out,
        "exact": exact_acc,
        "phase": phase_acc,
        "transformer": transformer_acc,
    }


@torch.no_grad()
def evaluate_contradictions(
    phase_model: StructuredPhaseModel,
    transformer: TinyTransformer,
    *,
    min_len: int = 3,
    max_len: int = 32,
    examples_per_length: int = 500,
    seed: int = 4321,
) -> Dict[str, List[float]]:
    """
    Each example is a loop. Half are forced to close (net phase 0).
    Half are forced inconsistent (net phase != 0).

    We ask each model for the net relation:
      predicted class 0 => "consistent"
      anything else    => "inconsistent"
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    phase_model.eval()
    transformer.eval()

    lengths_out, exact_acc, phase_acc, trans_acc = [], [], [], []

    for L in range(min_len, max_len + 1):
        B = examples_per_length
        tokens = torch.full((B, max_len), 4, dtype=torch.long, device=device)
        true_consistent = torch.zeros(B, dtype=torch.bool, device=device)

        for i in range(B):
            first = torch.randint(0, 4, (L - 1,), device=device)
            partial = int(first.sum().item()) % 4

            if i % 2 == 0:
                last = (-partial) % 4
                true_consistent[i] = True
            else:
                correct_last = (-partial) % 4
                last = (correct_last + int(torch.randint(1, 4, (1,), device=device))) % 4
                true_consistent[i] = False

            tokens[i, : L - 1] = first
            tokens[i, L - 1] = last

        exact_net = (tokens.clamp(max=3) * (tokens != 4).long()).sum(dim=1) % 4
        exact_pred_consistent = exact_net == 0

        phase_net = phase_model.predict_class(tokens)
        phase_pred_consistent = phase_net == 0

        trans_net = transformer(tokens).argmax(dim=1)
        trans_pred_consistent = trans_net == 0

        lengths_out.append(L)
        exact_acc.append((exact_pred_consistent == true_consistent).float().mean().item())
        phase_acc.append((phase_pred_consistent == true_consistent).float().mean().item())
        trans_acc.append((trans_pred_consistent == true_consistent).float().mean().item())

    return {
        "length": lengths_out,
        "exact": exact_acc,
        "phase": phase_acc,
        "transformer": trans_acc,
    }
