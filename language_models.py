from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


RELATION_PHRASES = {
    0: [
        "aligned with",
        "in the same orientation as",
        "pointing the same way as",
        "unrotated from",
    ],
    1: [
        "rotated ninety degrees clockwise from",
        "a quarter turn clockwise from",
        "turned right ninety degrees from",
        "one clockwise quarter turn from",
    ],
    2: [
        "opposite to",
        "rotated one hundred eighty degrees from",
        "a half turn from",
        "facing the reverse direction from",
    ],
    3: [
        "rotated ninety degrees counterclockwise from",
        "a quarter turn counterclockwise from",
        "turned left ninety degrees from",
        "one counterclockwise quarter turn from",
    ],
}

RELATION_LABELS = [
    "aligned / 0°",
    "90° clockwise",
    "opposite / 180°",
    "90° counterclockwise",
]


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    special = ["<pad>", "<sep>", "<cls>"]
    words = set()
    for variants in RELATION_PHRASES.values():
        for phrase in variants:
            words.update(simple_tokenize(phrase))
    vocab_list = special + sorted(words)
    word_to_id = {w: i for i, w in enumerate(vocab_list)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    return word_to_id, id_to_word


WORD_TO_ID, ID_TO_WORD = build_vocab()
PAD_ID = WORD_TO_ID["<pad>"]
SEP_ID = WORD_TO_ID["<sep>"]
CLS_ID = WORD_TO_ID["<cls>"]

TOKENIZED_PHRASES = {
    rel: [
        [WORD_TO_ID[w] for w in simple_tokenize(phrase)]
        for phrase in variants
    ]
    for rel, variants in RELATION_PHRASES.items()
}

MAX_PHRASE_TOKENS = max(
    len(tokens)
    for variants in TOKENIZED_PHRASES.values()
    for tokens in variants
)


def set_cpu_mode() -> None:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


@dataclass
class LanguageBatch:
    phrase_tokens: torch.Tensor   # [B, E, P]
    edge_mask: torch.Tensor       # [B, E]
    flat_tokens: torch.Tensor     # [B, S]
    targets: torch.Tensor         # [B]
    lengths: torch.Tensor         # [B]


def make_language_batch(
    batch_size: int,
    min_len: int,
    max_len: int,
    pad_edges_to: int,
    device: torch.device,
    rng: random.Random,
) -> LanguageBatch:
    phrase_tokens = torch.full(
        (batch_size, pad_edges_to, MAX_PHRASE_TOKENS),
        PAD_ID,
        dtype=torch.long,
        device=device,
    )
    edge_mask = torch.zeros(
        (batch_size, pad_edges_to),
        dtype=torch.bool,
        device=device,
    )

    # Flattened transformer sequence:
    # <cls> phrase <sep> phrase <sep> ...
    max_flat = 1 + pad_edges_to * (MAX_PHRASE_TOKENS + 1)
    flat_tokens = torch.full(
        (batch_size, max_flat),
        PAD_ID,
        dtype=torch.long,
        device=device,
    )
    flat_tokens[:, 0] = CLS_ID

    targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        L = rng.randint(min_len, max_len)
        lengths[b] = L
        rels = [rng.randrange(4) for _ in range(L)]
        targets[b] = sum(rels) % 4

        cursor = 1
        for e, rel in enumerate(rels):
            variants = TOKENIZED_PHRASES[rel]
            tokens = variants[rng.randrange(len(variants))]
            edge_mask[b, e] = True
            phrase_tokens[b, e, : len(tokens)] = torch.tensor(
                tokens, dtype=torch.long, device=device
            )

            end = cursor + len(tokens)
            flat_tokens[b, cursor:end] = torch.tensor(
                tokens, dtype=torch.long, device=device
            )
            cursor = end
            flat_tokens[b, cursor] = SEP_ID
            cursor += 1

    return LanguageBatch(
        phrase_tokens=phrase_tokens,
        edge_mask=edge_mask,
        flat_tokens=flat_tokens,
        targets=targets,
        lengths=lengths,
    )


class LanguagePhaseModel(nn.Module):
    """
    Learns language -> unit complex relation.

    It is NOT told which phrase means 0, +90, 180, or -90.
    It only gets final path labels during training.

    Structural inductive bias:
        each edge is mapped to a unit complex number;
        path relations compose by complex multiplication.
    """

    def __init__(self, vocab_size: int, d_model: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def encode_phrases(self, phrase_tokens: torch.Tensor) -> torch.Tensor:
        # [B,E,P,D]
        emb = self.embedding(phrase_tokens)
        tok_mask = (phrase_tokens != PAD_ID).unsqueeze(-1)
        summed = (emb * tok_mask).sum(dim=2)
        counts = tok_mask.sum(dim=2).clamp(min=1)
        pooled = summed / counts
        xy = self.encoder(pooled)
        xy = xy / xy.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return xy

    def compose(self, edge_xy: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        B, E, _ = edge_xy.shape
        result = torch.zeros(B, 2, dtype=edge_xy.dtype, device=edge_xy.device)
        result[:, 0] = 1.0  # identity = 1 + 0i

        identity = torch.zeros_like(edge_xy)
        identity[..., 0] = 1.0
        edges = torch.where(edge_mask.unsqueeze(-1), edge_xy, identity)

        for e in range(E):
            a = result
            b = edges[:, e, :]
            real = a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1]
            imag = a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]
            result = torch.stack((real, imag), dim=1)
            result = result / result.norm(dim=1, keepdim=True).clamp(min=1e-8)

        return result

    def forward(self, phrase_tokens: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        edge_xy = self.encode_phrases(phrase_tokens)
        final_xy = self.compose(edge_xy, edge_mask)
        canonical = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=final_xy.dtype,
            device=final_xy.device,
        )
        # Scale cosine similarities so CE has useful gradients.
        return 10.0 * (final_xy @ canonical.T)

    @torch.no_grad()
    def phrase_phase_table(self) -> List[dict]:
        rows = []
        device = next(self.parameters()).device
        for rel, variants in RELATION_PHRASES.items():
            for idx, phrase in enumerate(variants):
                ids = TOKENIZED_PHRASES[rel][idx]
                x = torch.full(
                    (1, 1, MAX_PHRASE_TOKENS),
                    PAD_ID,
                    dtype=torch.long,
                    device=device,
                )
                x[0, 0, : len(ids)] = torch.tensor(ids, device=device)
                xy = self.encode_phrases(x)[0, 0]
                angle = float(torch.atan2(xy[1], xy[0]).cpu())
                predicted = int(
                    torch.argmax(
                        torch.tensor([
                            xy[0],
                            xy[1],
                            -xy[0],
                            -xy[1],
                        ])
                    ).item()
                )
                rows.append(
                    {
                        "phrase": phrase,
                        "true_relation": rel,
                        "predicted_relation": predicted,
                        "angle_radians": angle,
                    }
                )
        return rows


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class LanguageTransformer(nn.Module):
    """
    Generic sequence baseline over the same relation phrases.

    It sees the same words, but is not given the complex composition rule.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dim_feedforward: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = SinusoidalPositionEncoding(d_model, max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, 4)

    def forward(self, flat_tokens: torch.Tensor) -> torch.Tensor:
        pad_mask = flat_tokens == PAD_ID
        x = self.embedding(flat_tokens)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(x[:, 0, :])


def train_language_models(
    *,
    seed: int = 11,
    train_max_len: int = 5,
    test_max_len: int = 32,
    steps: int = 900,
    batch_size: int = 128,
    progress_callback=None,
):
    set_cpu_mode()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    device = torch.device("cpu")

    vocab_size = len(WORD_TO_ID)
    max_flat = 1 + test_max_len * (MAX_PHRASE_TOKENS + 1)

    phase_model = LanguagePhaseModel(vocab_size).to(device)
    transformer = LanguageTransformer(vocab_size, max_flat).to(device)

    phase_opt = torch.optim.AdamW(phase_model.parameters(), lr=3e-3, weight_decay=1e-4)
    trans_opt = torch.optim.AdamW(transformer.parameters(), lr=2e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    history = {"phase_loss": [], "transformer_loss": []}

    for step in range(1, steps + 1):
        batch = make_language_batch(
            batch_size=batch_size,
            min_len=1,
            max_len=train_max_len,
            pad_edges_to=test_max_len,
            device=device,
            rng=rng,
        )

        phase_opt.zero_grad(set_to_none=True)
        phase_logits = phase_model(batch.phrase_tokens, batch.edge_mask)
        phase_loss = ce(phase_logits, batch.targets)
        phase_loss.backward()
        torch.nn.utils.clip_grad_norm_(phase_model.parameters(), 1.0)
        phase_opt.step()

        trans_opt.zero_grad(set_to_none=True)
        trans_logits = transformer(batch.flat_tokens)
        trans_loss = ce(trans_logits, batch.targets)
        trans_loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        trans_opt.step()

        history["phase_loss"].append(float(phase_loss.detach().cpu()))
        history["transformer_loss"].append(float(trans_loss.detach().cpu()))

        if progress_callback and (step == 1 or step % 25 == 0 or step == steps):
            progress_callback(
                step,
                steps,
                float(phase_loss.detach().cpu()),
                float(trans_loss.detach().cpu()),
            )

    return phase_model, transformer, history


@torch.no_grad()
def evaluate_language_by_length(
    phase_model: LanguagePhaseModel,
    transformer: LanguageTransformer,
    *,
    min_len: int,
    max_len: int,
    examples_per_length: int,
    seed: int,
):
    device = torch.device("cpu")
    phase_model.eval()
    transformer.eval()
    rng = random.Random(seed)

    out = {
        "length": [],
        "phase": [],
        "transformer": [],
    }

    for L in range(min_len, max_len + 1):
        batch = make_language_batch(
            batch_size=examples_per_length,
            min_len=L,
            max_len=L,
            pad_edges_to=max_len,
            device=device,
            rng=rng,
        )
        phase_pred = phase_model(batch.phrase_tokens, batch.edge_mask).argmax(dim=1)
        trans_pred = transformer(batch.flat_tokens).argmax(dim=1)

        out["length"].append(L)
        out["phase"].append(
            float((phase_pred == batch.targets).float().mean().cpu())
        )
        out["transformer"].append(
            float((trans_pred == batch.targets).float().mean().cpu())
        )

    return out


@torch.no_grad()
def evaluate_language_contradictions(
    phase_model: LanguagePhaseModel,
    transformer: LanguageTransformer,
    *,
    min_len: int,
    max_len: int,
    examples_per_length: int,
    seed: int,
):
    """
    Construct language loops. Half close to identity; half do not.
    Use the already-trained net-relation predictor:
      class 0 => consistent
      class 1/2/3 => inconsistent
    """
    device = torch.device("cpu")
    phase_model.eval()
    transformer.eval()
    rng = random.Random(seed)

    results = {"length": [], "phase": [], "transformer": []}

    for L in range(min_len, max_len + 1):
        B = examples_per_length
        phrase_tokens = torch.full(
            (B, max_len, MAX_PHRASE_TOKENS),
            PAD_ID,
            dtype=torch.long,
            device=device,
        )
        edge_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)
        max_flat = 1 + max_len * (MAX_PHRASE_TOKENS + 1)
        flat_tokens = torch.full((B, max_flat), PAD_ID, dtype=torch.long, device=device)
        flat_tokens[:, 0] = CLS_ID
        true_consistent = torch.zeros(B, dtype=torch.bool, device=device)

        for b in range(B):
            rels = [rng.randrange(4) for _ in range(L - 1)]
            partial = sum(rels) % 4

            if b % 2 == 0:
                last = (-partial) % 4
                true_consistent[b] = True
            else:
                correct = (-partial) % 4
                last = (correct + rng.choice([1, 2, 3])) % 4
                true_consistent[b] = False

            rels.append(last)
            cursor = 1
            for e, rel in enumerate(rels):
                variants = TOKENIZED_PHRASES[rel]
                ids = variants[rng.randrange(len(variants))]
                edge_mask[b, e] = True
                phrase_tokens[b, e, :len(ids)] = torch.tensor(ids, device=device)

                end = cursor + len(ids)
                flat_tokens[b, cursor:end] = torch.tensor(ids, device=device)
                cursor = end
                flat_tokens[b, cursor] = SEP_ID
                cursor += 1

        phase_net = phase_model(phrase_tokens, edge_mask).argmax(dim=1)
        trans_net = transformer(flat_tokens).argmax(dim=1)

        phase_consistent = phase_net == 0
        trans_consistent = trans_net == 0

        results["length"].append(L)
        results["phase"].append(
            float((phase_consistent == true_consistent).float().mean().cpu())
        )
        results["transformer"].append(
            float((trans_consistent == true_consistent).float().mean().cpu())
        )

    return results
