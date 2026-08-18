"""Experiment 1b models.

Two systems, same data, same labels.

RELATIONAL
    A character encoder reads a *pair* of adjacent word forms and resolves it
    into a relation state. Relation states then compose, step by step, through
    a learned binary operator, exactly as in Experiment 3. Depth costs
    iterations, not architecture.

TRANSFORMER
    Sees the whole chain as one flat character sequence and predicts the
    composed relation directly. Fixed depth regardless of chain length.

Nothing anywhere has access to word meaning. Input is characters.
"""
from __future__ import annotations

import copy
import math
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from morphology_data import (
    CATEGORIES,
    CAT_INDEX,
    Chain,
    CharVocabulary,
    NUM_COMPOSED_RELATIONS,
    OPERATIONS,
)

COMMITMENT_WEIGHT = 0.25


def cpu_setup():
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def parameter_count(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def encode_pair_batch(chains: List[Chain], vocab: CharVocabulary,
                      max_depth=6, max_chars=24, device=torch.device("cpu")):
    """Each chain becomes a sequence of (source, result) character pairs."""
    B = len(chains)
    left = torch.zeros(B, max_depth, max_chars, dtype=torch.long, device=device)
    right = torch.zeros(B, max_depth, max_chars, dtype=torch.long, device=device)
    mask = torch.zeros(B, max_depth, dtype=torch.bool, device=device)
    for b, c in enumerate(chains):
        for i, (a, z) in enumerate(c.pairs[:max_depth]):
            left[b, i] = torch.tensor(vocab.encode(a, max_chars), device=device)
            right[b, i] = torch.tensor(vocab.encode(z, max_chars), device=device)
            mask[b, i] = True
    targets = {
        "composed": torch.tensor([c.composed_relation() for c in chains],
                                 dtype=torch.long, device=device),
        # The two group factors, predicted separately. Z3 x Z2 is a product, so
        # a model that has learned the group can emit a (cycle, polarity)
        # combination it never saw labeled. A single 6-way head cannot.
        "cycle": torch.tensor([c.cycle_advance % 3 for c in chains],
                              dtype=torch.long, device=device),
        "polarity": torch.tensor([c.polarity for c in chains],
                                 dtype=torch.long, device=device),
    }
    return left, right, mask, targets


def encode_flat_batch(chains: List[Chain], vocab: CharVocabulary,
                      max_tokens=180, device=torch.device("cpu")):
    B = len(chains)
    x = torch.zeros(B, max_tokens, dtype=torch.long, device=device)
    sep = 0
    for b, c in enumerate(chains):
        ids: List[int] = []
        for f in c.forms:
            ids.extend(vocab.encode(f, 24)[: len(f)])
            ids.append(sep)
        ids = ids[:max_tokens]
        x[b, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    targets = {
        "composed": torch.tensor([c.composed_relation() for c in chains],
                                 dtype=torch.long, device=device),
        "cycle": torch.tensor([c.cycle_advance % 3 for c in chains],
                              dtype=torch.long, device=device),
        "polarity": torch.tensor([c.polarity for c in chains],
                                 dtype=torch.long, device=device),
    }
    return x, targets


# ---------------------------------------------------------------------------
# Relational model
# ---------------------------------------------------------------------------

class CharFormEncoder(nn.Module):
    """Characters -> one vector per word form. No lexicon, no embeddings."""

    def __init__(self, vocab_size, char_dim=32, form_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_dim, padding_idx=0)
        self.gru = nn.GRU(char_dim, form_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, chars):
        B, S, C = chars.shape
        flat = chars.reshape(B * S, C)
        emb = self.embedding(flat)
        out, _ = self.gru(emb)
        m = (flat != 0).unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp_min(1)
        return pooled.reshape(B, S, -1)


class RelationResolver(nn.Module):
    """A (source, result) form pair -> a relation state.

    This is the step Experiment 1 tested with phrases. Here the surface signal
    is the morphological difference between two forms, including cases where
    there is no difference at all (German zero-plural) and cases where the two
    forms share no characters (go/went).
    """

    def __init__(self, form_dim=64, state_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(form_dim * 3, 96),
            nn.GELU(),
            nn.Linear(96, state_dim),
        )

    def forward(self, left, right):
        delta = right - left
        return F.normalize(self.net(torch.cat([left, right, delta], dim=-1)), dim=-1)


class CompositionOperator(nn.Module):
    """Learned binary operator over relation states (Experiment-3 lineage)."""

    def __init__(self, state_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, state_dim),
        )

    def forward(self, acc, nxt):
        return F.normalize(self.net(torch.cat([acc, nxt], dim=-1)), dim=-1)


class AngleResolver(nn.Module):
    """A (source, result) form pair -> a vector of angles.

    The alternative to a free-vector relation state. Each relation is a point
    on a set of circles, so composing two relations is adding their angles.
    Closure, associativity and inversion are properties of addition rather than
    things a network has to induce from examples.

    This matters for extrapolation. A learned operator trained only on chains
    of length one and two never observes a case where a count exceeds the
    modulus and has to wrap, so it cannot distinguish modular arithmetic from
    plain counting. Angles wrap because that is what angles do.
    """

    def __init__(self, form_dim=64, n_angles=12):
        super().__init__()
        self.n_angles = n_angles
        self.net = nn.Sequential(
            nn.Linear(form_dim * 3, 96),
            nn.GELU(),
            nn.Linear(96, n_angles),
        )

    def forward(self, left, right):
        delta = right - left
        return self.net(torch.cat([left, right, delta], dim=-1))   # raw angles, radians


class RelationCodebook(nn.Module):
    """Optional closure: snap composed states onto a learned finite set.

    Unlike Experiment 5, snapping here is applied to *complete relation
    states*, never to a partially-read description. Every intermediate value in
    this model is a genuine composed relation, which is the precondition the
    Experiment 5 architecture could not satisfy.
    """

    def __init__(self, state_dim, num_states=64):
        super().__init__()
        self.num_states = num_states
        g = torch.Generator().manual_seed(0xC0DEB00C)
        self.states = nn.Parameter(F.normalize(torch.randn(num_states, state_dim, generator=g), dim=-1))

    def forward(self, h):
        h_n = F.normalize(h, dim=-1)
        book = F.normalize(self.states, dim=-1)
        nearest = book[(h_n @ book.t()).argmax(dim=-1)]
        commitment = F.mse_loss(h_n, nearest.detach()) + F.mse_loss(nearest, h_n.detach())
        return h_n + (nearest - h_n).detach(), commitment


class RelationalChainModel(nn.Module):
    """composition="learned"  -> free relation vectors, MLP operator
       composition="additive" -> angular relation states, composition is addition
    """

    def __init__(self, vocab_size, char_dim=32, form_dim=64, state_dim=48,
                 cleanup=True, num_states=64, composition="additive", n_angles=12):
        super().__init__()
        self.composition = composition
        self.encoder = CharFormEncoder(vocab_size, char_dim, form_dim)
        self.state_dim = state_dim
        self.n_angles = n_angles

        if composition == "additive":
            self.resolver = AngleResolver(form_dim, n_angles)
            self.operator = None
            self.identity = None
            self.cleanup = False          # additive closure is already exact
            self.codebook = None
            head_dim = 2 * n_angles       # cos and sin of each accumulated angle
        else:
            self.resolver = RelationResolver(form_dim, state_dim)
            self.operator = CompositionOperator(state_dim)
            self.identity = nn.Parameter(F.normalize(torch.randn(state_dim), dim=0))
            self.cleanup = cleanup
            self.codebook = RelationCodebook(state_dim, num_states) if cleanup else None
            head_dim = state_dim

        self.last_commitment = torch.zeros(())
        # Factored heads over the two group generators.
        self.cycle_head = nn.Linear(head_dim, 3)
        self.polarity_head = nn.Linear(head_dim, 2)

    def forward(self, left, right, mask):
        B, S, _ = left.shape
        lv = self.encoder(left)
        rv = self.encoder(right)
        steps = self.resolver(lv, rv)

        if self.composition == "additive":
            # Sum the per-step angles. Masked-out steps contribute zero, which
            # is the identity of the group, so padding is harmless by
            # construction rather than by special-casing.
            acc = (steps * mask.unsqueeze(-1).to(steps.dtype)).sum(dim=1)
            feats = torch.cat([torch.cos(acc), torch.sin(acc)], dim=-1)
            self.last_commitment = torch.zeros((), device=left.device)
            return {"cycle": self.cycle_head(feats),
                    "polarity": self.polarity_head(feats)}

        acc = F.normalize(self.identity, dim=0).unsqueeze(0).expand(B, -1)
        commitment = torch.zeros((), device=left.device)
        n = 0
        for i in range(S):
            active = mask[:, i]
            if not active.any():
                break
            nxt = self.operator(acc, steps[:, i])
            if self.codebook is not None:
                nxt, c = self.codebook(nxt)
                commitment = commitment + c
                n += 1
            acc = torch.where(active.unsqueeze(-1), nxt, acc)
        self.last_commitment = commitment / max(n, 1)
        return {"cycle": self.cycle_head(acc), "polarity": self.polarity_head(acc)}

    def encoder_shell(self, new_vocab_size):
        """Freeze the relation machinery, give it a fresh character encoder.

        This is the cross-linguistic test: the composition operator, the
        codebook and the output head all carry over untouched. Only the part
        that reads characters is allowed to adapt to the new language.
        """
        shell = RelationalChainModel(
            new_vocab_size,
            char_dim=self.encoder.embedding.embedding_dim,
            form_dim=self.resolver.net[0].in_features // 3,
            state_dim=self.state_dim,
            cleanup=self.cleanup,
            num_states=(self.codebook.num_states if self.codebook is not None else 64),
            composition=self.composition,
            n_angles=self.n_angles,
        )
        shell.resolver.load_state_dict(copy.deepcopy(self.resolver.state_dict()))
        shell.cycle_head.load_state_dict(copy.deepcopy(self.cycle_head.state_dict()))
        shell.polarity_head.load_state_dict(copy.deepcopy(self.polarity_head.state_dict()))
        frozen = [shell.resolver, shell.cycle_head, shell.polarity_head]

        if self.operator is not None:
            shell.operator.load_state_dict(copy.deepcopy(self.operator.state_dict()))
            frozen.append(shell.operator)
        if self.identity is not None:
            shell.identity.data.copy_(self.identity.data)
            shell.identity.requires_grad = False
        if self.codebook is not None:
            shell.codebook.load_state_dict(copy.deepcopy(self.codebook.state_dict()))
            frozen.append(shell.codebook)

        for module in frozen:
            for p in module.parameters():
                p.requires_grad = False
        return shell

    def scrambled_shell(self, new_vocab_size, seed):
        """Control: same transferred weights, composition semantics destroyed.

        Randomizes the operator's output layer rather than permuting it, so no
        downstream trainable map can undo the damage. (Experiment 4's control
        used a permutation, which a trainable head learns around for free.)
        """
        shell = self.encoder_shell(new_vocab_size)
        g = torch.Generator().manual_seed(seed)
        # Under additive composition there is no operator to damage — the
        # learned content lives entirely in the resolver, so that is what gets
        # randomized. Either way the target is the transferred relation
        # machinery, not the character reader.
        target = shell.operator if shell.operator is not None else shell.resolver
        with torch.no_grad():
            last = target.net[-1]
            std = last.weight.std().item()
            last.weight.copy_(torch.randn(last.weight.shape, generator=g) * std)
            last.bias.zero_()
        for p in target.parameters():
            p.requires_grad = False
        return shell


# ---------------------------------------------------------------------------
# Transformer baseline
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=180):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TinyChainTransformer(nn.Module):
    def __init__(self, vocab_size, max_tokens=180, d_model=64, layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_tokens)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.cycle_head = nn.Linear(d_model, 3)
        self.polarity_head = nn.Linear(d_model, 2)

    def forward(self, tokens):
        pad = tokens == 0
        x = self.pos(self.embedding(tokens))
        x = self.encoder(x, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (x * keep).sum(1) / keep.sum(1).clamp_min(1)
        return {"cycle": self.cycle_head(pooled), "polarity": self.polarity_head(pooled)}


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def _sample(ds, batch, rng):
    return [ds[rng.randrange(len(ds))] for _ in range(batch)]


def train_relational(model, ds, vocab, steps, batch_size, lr, seed, progress=None):
    cpu_setup()
    model.train()
    pars = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(pars, lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    ce = nn.CrossEntropyLoss()
    for step in range(1, steps + 1):
        b = _sample(ds, batch_size, rng)
        left, right, mask, y = encode_pair_batch(b, vocab)
        opt.zero_grad(set_to_none=True)
        out = model(left, right, mask)
        loss = ce(out["cycle"], y["cycle"]) + ce(out["polarity"], y["polarity"])
        if model.codebook is not None:
            loss = loss + COMMITMENT_WEIGHT * model.last_commitment
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pars, 1.0)
        opt.step()
        if progress and (step == 1 or step % 25 == 0 or step == steps):
            both = ((out["cycle"].argmax(1) == y["cycle"]) &
                    (out["polarity"].argmax(1) == y["polarity"]))
            progress(step, steps, float(loss.detach()), float(both.float().mean()))
    model.eval()
    return model


def train_transformer(model, ds, vocab, steps, batch_size, lr, seed, progress=None):
    cpu_setup()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    ce = nn.CrossEntropyLoss()
    for step in range(1, steps + 1):
        b = _sample(ds, batch_size, rng)
        x, y = encode_flat_batch(b, vocab)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = ce(out["cycle"], y["cycle"]) + ce(out["polarity"], y["polarity"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if progress and (step == 1 or step % 25 == 0 or step == steps):
            both = ((out["cycle"].argmax(1) == y["cycle"]) &
                    (out["polarity"].argmax(1) == y["polarity"]))
            progress(step, steps, float(loss.detach()), float(both.float().mean()))
    model.eval()
    return model


def _both_correct(out, y):
    return ((out["cycle"].argmax(1) == y["cycle"]) &
            (out["polarity"].argmax(1) == y["polarity"]))


@torch.no_grad()
def eval_relational(model, ds, vocab, batch_size=128):
    """Scored on the full composed relation: both group factors must be right."""
    model.eval()
    correct = total = 0
    for i in range(0, len(ds), batch_size):
        b = ds[i:i + batch_size]
        left, right, mask, y = encode_pair_batch(b, vocab)
        correct += int(_both_correct(model(left, right, mask), y).sum())
        total += len(b)
    return correct / max(total, 1)


@torch.no_grad()
def eval_transformer(model, ds, vocab, batch_size=128):
    model.eval()
    correct = total = 0
    for i in range(0, len(ds), batch_size):
        b = ds[i:i + batch_size]
        x, y = encode_flat_batch(b, vocab)
        correct += int(_both_correct(model(x), y).sum())
        total += len(b)
    return correct / max(total, 1)


def majority_baseline(train_ds, test_ds):
    """Chance floor: independently predict the most common value of each factor.

    Matched to the factored heads, so it is the honest floor for this task
    rather than a joint-class floor the models are not optimizing.
    """
    from collections import Counter
    if not train_ds or not test_ds:
        return 0.0
    top_cycle = Counter(c.cycle_advance % 3 for c in train_ds).most_common(1)[0][0]
    top_pol = Counter(c.polarity for c in train_ds).most_common(1)[0][0]
    hits = sum(1 for c in test_ds
               if c.cycle_advance % 3 == top_cycle and c.polarity == top_pol)
    return hits / len(test_ds)


def unseen_combination_rate(train_ds, test_ds):
    """Share of test chains whose (cycle, polarity) pair never appeared in training.

    These are only solvable by composing the two factors independently. A model
    that memorizes joint classes scores zero on them.
    """
    if not test_ds:
        return 0.0
    seen = {(c.cycle_advance % 3, c.polarity) for c in train_ds}
    miss = sum(1 for c in test_ds if (c.cycle_advance % 3, c.polarity) not in seen)
    return miss / len(test_ds)
