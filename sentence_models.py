"""Experiment 6 models — sentence-level relation induction.

The resolver reads a sentence as characters and is told which two spans are the
entities. It must output the relation that runs from the first to the second.

Direction is the point of the span markers. "A causes B" and "B is caused by A"
are the same relation with the surface order reversed, so a model that reads
position rather than structure gets exactly one of those backwards. Feeding the
difference of the two span vectors gives the network an antisymmetric term to
work with; whether it uses it is what T2 measures.

Angles again, for the same reason as Experiment 1b: polarity is a Z2 group and
composing relations means adding polarities. Kind is a five-way classification
and does not compose, so it gets an ordinary head.
"""
from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from relational_lexicon import KINDS
from sentence_data import CharVocabulary, Sample

MAX_CHARS = 160


def cpu_setup():
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def parameter_count(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


def _span_offsets(text: str, span: str) -> Tuple[int, int]:
    """Character range of a span inside the sentence. (-1,-1) if absent."""
    i = text.lower().find(span.lower())
    if i < 0:
        return (-1, -1)
    return (i, min(i + len(span), MAX_CHARS))


def encode_batch(samples: Sequence[Sample], vocab: CharVocabulary,
                 device=torch.device("cpu")):
    B = len(samples)
    chars = torch.zeros(B, MAX_CHARS, dtype=torch.long, device=device)
    a_mask = torch.zeros(B, MAX_CHARS, device=device)
    b_mask = torch.zeros(B, MAX_CHARS, device=device)
    for i, s in enumerate(samples):
        chars[i] = torch.tensor(vocab.encode(s.text, MAX_CHARS), device=device)
        for span, m in ((s.source, a_mask), (s.target, b_mask)):
            lo, hi = _span_offsets(s.text, span)
            if lo >= 0:
                m[i, lo:hi] = 1.0
    targets = {
        "kind": torch.tensor([s.kind for s in samples], dtype=torch.long, device=device),
        "polarity": torch.tensor([s.polarity for s in samples], dtype=torch.long, device=device),
    }
    return chars, a_mask, b_mask, targets


class SentenceResolver(nn.Module):
    def __init__(self, vocab_size, char_dim=32, hidden=96, n_angles=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_dim, padding_idx=0)
        self.gru = nn.GRU(char_dim, hidden // 2, batch_first=True, bidirectional=True)
        self.n_angles = n_angles
        self.to_angles = nn.Sequential(
            nn.Linear(hidden * 4, 128),
            nn.GELU(),
            nn.Linear(128, n_angles),
        )
        self.kind_head = nn.Linear(2 * n_angles, len(KINDS))
        self.polarity_head = nn.Linear(2 * n_angles, 2)

    def forward(self, chars, a_mask, b_mask):
        emb = self.embedding(chars)
        ctx, _ = self.gru(emb)                                  # (B, L, H)
        pad = (chars != 0).unsqueeze(-1).float()
        sent = (ctx * pad).sum(1) / pad.sum(1).clamp_min(1)
        a = (ctx * a_mask.unsqueeze(-1)).sum(1) / a_mask.sum(1, keepdim=True).clamp_min(1)
        b = (ctx * b_mask.unsqueeze(-1)).sum(1) / b_mask.sum(1, keepdim=True).clamp_min(1)
        # a - b is antisymmetric: swapping the spans negates it, which is what
        # lets the model represent direction at all.
        feats = torch.cat([sent, a, b, a - b], dim=-1)
        angles = self.to_angles(feats)
        h = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return {"kind": self.kind_head(h), "polarity": self.polarity_head(h),
                "angles": angles}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_CHARS):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TinySentenceTransformer(nn.Module):
    """Same inputs, same labels. Span markers are added to the embedding so it
    is not handicapped on direction."""

    def __init__(self, vocab_size, d_model=64, layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.span_a = nn.Parameter(torch.randn(d_model) * 0.02)
        self.span_b = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.kind_head = nn.Linear(d_model, len(KINDS))
        self.polarity_head = nn.Linear(d_model, 2)

    def forward(self, chars, a_mask, b_mask):
        pad = chars == 0
        x = self.embedding(chars)
        x = x + a_mask.unsqueeze(-1) * self.span_a + b_mask.unsqueeze(-1) * self.span_b
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (x * keep).sum(1) / keep.sum(1).clamp_min(1)
        return {"kind": self.kind_head(pooled), "polarity": self.polarity_head(pooled)}


def _sample(ds, batch, rng):
    return [ds[rng.randrange(len(ds))] for _ in range(batch)]


def train_model(model, ds, vocab, steps, batch_size, lr, seed, progress=None):
    cpu_setup()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    ce = nn.CrossEntropyLoss()
    for step in range(1, steps + 1):
        b = _sample(ds, batch_size, rng)
        chars, am, bm, y = encode_batch(b, vocab)
        opt.zero_grad(set_to_none=True)
        out = model(chars, am, bm)
        loss = ce(out["kind"], y["kind"]) + ce(out["polarity"], y["polarity"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if progress and (step == 1 or step % 25 == 0 or step == steps):
            both = ((out["kind"].argmax(1) == y["kind"]) &
                    (out["polarity"].argmax(1) == y["polarity"]))
            progress(step, steps, float(loss.detach()), float(both.float().mean()))
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, ds, vocab, batch_size=128):
    """Both factors correct, plus each factor separately."""
    model.eval()
    both = kind_ok = pol_ok = total = 0
    for i in range(0, len(ds), batch_size):
        b = ds[i:i + batch_size]
        chars, am, bm, y = encode_batch(b, vocab)
        out = model(chars, am, bm)
        k = out["kind"].argmax(1) == y["kind"]
        p = out["polarity"].argmax(1) == y["polarity"]
        both += int((k & p).sum())
        kind_ok += int(k.sum())
        pol_ok += int(p.sum())
        total += len(b)
    n = max(total, 1)
    return {"both": both / n, "kind": kind_ok / n, "polarity": pol_ok / n, "n": total}


@torch.no_grad()
def _predict(model, ds: Sequence[Sample], vocab, batch_size=128):
    ks, ps = [], []
    for i in range(0, len(ds), batch_size):
        chars, am, bm, _ = encode_batch(list(ds[i:i + batch_size]), vocab)
        out = model(chars, am, bm)
        ks.append(out["kind"].argmax(1))
        ps.append(out["polarity"].argmax(1))
    if not ks:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.cat(ks), torch.cat(ps)


@torch.no_grad()
def evaluate_chains(model, chains, vocab, batch_size=128):
    """Read each sentence separately, then compose by addition.

    The model never sees the two sentences together and is never trained on a
    chain. Polarity is XOR'd; kind must come out the same from both halves.
    This is the sentence-level form of the composition that carried
    Experiment 1b past its training depth.
    """
    model.eval()
    if not chains:
        return {"polarity": 0.0, "kind": 0.0, "both": 0.0, "n": 0}
    k1, p1 = _predict(model, [c.first for c in chains], vocab, batch_size)
    k2, p2 = _predict(model, [c.second for c in chains], vocab, batch_size)
    true_k = torch.tensor([c.kind for c in chains], dtype=torch.long)
    true_p = torch.tensor([c.polarity for c in chains], dtype=torch.long)
    pol_ok = (p1 ^ p2) == true_p
    kind_ok = (k1 == true_k) & (k2 == true_k)
    n = len(chains)
    return {"polarity": float(pol_ok.float().mean()),
            "kind": float(kind_ok.float().mean()),
            "both": float((pol_ok & kind_ok).float().mean()),
            "n": n}


def majority_floor(train_ds: List[Sample], test_ds: List[Sample]) -> float:
    from collections import Counter
    if not train_ds or not test_ds:
        return 0.0
    top = Counter((s.kind, s.polarity) for s in train_ds).most_common(1)[0][0]
    hits = sum(1 for s in test_ds if (s.kind, s.polarity) == top)
    return hits / len(test_ds)
