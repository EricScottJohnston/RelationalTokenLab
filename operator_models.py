from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


RELATION_COUNT = 4
UNKNOWN = 4
RELATION_NAMES = ["0 / identity", "+1 quarter-turn", "+2 / opposite", "-1 quarter-turn", "UNKNOWN"]


def set_cpu_mode() -> None:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def ground_truth_compose(rels: List[int]) -> int:
    """Hidden world law used only by the data generator / evaluator."""
    return int(sum(int(r) for r in rels) % 4)


def make_sequence_batch(
    batch_size: int,
    min_len: int,
    max_len: int,
    pad_to: int,
    device: torch.device,
    rng: random.Random,
):
    tokens = torch.full((batch_size, pad_to), UNKNOWN, dtype=torch.long, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b in range(batch_size):
        L = rng.randint(min_len, max_len)
        rels = [rng.randrange(4) for _ in range(L)]
        tokens[b, :L] = torch.tensor(rels, dtype=torch.long, device=device)
        lengths[b] = L
        targets[b] = ground_truth_compose(rels)

    return tokens, lengths, targets


class LearnedBinaryOperator(nn.Module):
    """
    Learns relation embeddings AND the binary composition operator.

    Important: there is no complex multiplication and no modular addition
    anywhere in this model.

    Relation 0 is designated as the identity only for the STRUCTURED variant's
    regularizers. The operator itself is a neural MLP.
    """

    def __init__(self, latent_dim: int = 8, structured: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.structured = structured

        self.raw_embeddings = nn.Parameter(torch.randn(RELATION_COUNT, latent_dim) * 0.4)

        self.operator = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )

        # Learned scalar temperature for prototype classification.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))

    def prototypes(self) -> torch.Tensor:
        return F.normalize(self.raw_embeddings, dim=-1)

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.prototypes()[ids]

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = self.operator(torch.cat([a, b], dim=-1))
        return F.normalize(out, dim=-1)

    def classify_latent(self, z: torch.Tensor) -> torch.Tensor:
        proto = self.prototypes()
        scale = self.logit_scale.exp().clamp(1.0, 50.0)
        return scale * (z @ proto.T)

    def fold(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Recursive composition. Start at first relation, then repeatedly apply
        the learned binary operator.
        """
        B, S = tokens.shape
        proto = self.prototypes()

        # Length is always >= 1 in this experiment.
        z = proto[tokens[:, 0].clamp(max=3)]

        for pos in range(1, S):
            active = lengths > pos
            if not active.any():
                break

            nxt = proto[tokens[:, pos].clamp(max=3)]
            composed = self.op(z, nxt)
            z = torch.where(active.unsqueeze(-1), composed, z)

        return z

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classify_latent(self.fold(tokens, lengths))

    def structural_losses(self) -> Dict[str, torch.Tensor]:
        """
        Only called for the STRUCTURED model.

        These constraints do NOT specify the composition table:
          - identity behavior
          - inverse consistency
          - associativity
          - closure onto the finite relation-state set

        Inverse pairs are supplied as a structural relation:
          inv(0)=0, inv(1)=3, inv(2)=2, inv(3)=1.
        """
        proto = self.prototypes()
        e = proto[0]

        # Identity
        left = self.op(e.expand_as(proto), proto)
        right = self.op(proto, e.expand_as(proto))
        identity = F.mse_loss(left, proto) + F.mse_loss(right, proto)

        # Inverse consistency. We give inverse pairings but not their general
        # composition table.
        inv_ids = torch.tensor([0, 3, 2, 1], device=proto.device)
        inv_proto = proto[inv_ids]
        inv_left = self.op(proto, inv_proto)
        inv_right = self.op(inv_proto, proto)
        inverse = F.mse_loss(inv_left, e.expand_as(inv_left)) + F.mse_loss(
            inv_right, e.expand_as(inv_right)
        )

        # Associativity over all 4^3 primitive triples.
        triples = torch.cartesian_prod(
            torch.arange(4, device=proto.device),
            torch.arange(4, device=proto.device),
            torch.arange(4, device=proto.device),
        )
        a = proto[triples[:, 0]]
        b = proto[triples[:, 1]]
        c = proto[triples[:, 2]]
        lhs = self.op(self.op(a, b), c)
        rhs = self.op(a, self.op(b, c))
        associativity = F.mse_loss(lhs, rhs)

        # Closure: outputs of all primitive pairs should land near one of the
        # four learned relation prototypes, without specifying WHICH one.
        pairs = torch.cartesian_prod(
            torch.arange(4, device=proto.device),
            torch.arange(4, device=proto.device),
        )
        pair_out = self.op(proto[pairs[:, 0]], proto[pairs[:, 1]])
        d2 = ((pair_out[:, None, :] - proto[None, :, :]) ** 2).sum(dim=-1)
        tau = 0.08
        softmin = -tau * torch.logsumexp(-d2 / tau, dim=1)
        closure = softmin.mean()

        # Keep prototypes distinct enough to avoid trivial collapse.
        sim = proto @ proto.T
        offdiag = sim[~torch.eye(4, dtype=torch.bool, device=proto.device)]
        separation = torch.relu(offdiag - 0.75).pow(2).mean()

        return {
            "identity": identity,
            "inverse": inverse,
            "associativity": associativity,
            "closure": closure,
            "separation": separation,
        }


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

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TinyTransformer(nn.Module):
    """
    Generic sequence baseline. It receives the same primitive relation IDs but
    no recursive learned operator and no structural constraints.
    """

    def __init__(self, max_len: int, d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        # 0..3 relations, 4 PAD, 5 CLS
        self.embedding = nn.Embedding(6, d_model)
        self.pos = SinusoidalPositionEncoding(d_model, max_len + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.size(0)
        cls = torch.full((B, 1), 5, dtype=torch.long, device=tokens.device)
        seq = torch.cat([cls, tokens], dim=1)
        pad_mask = seq == UNKNOWN
        x = self.embedding(seq)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(x[:, 0])


def train_models(
    *,
    seed: int = 29,
    train_max_len: int = 5,
    test_max_len: int = 64,
    steps: int = 950,
    batch_size: int = 160,
    progress_callback=None,
):
    set_cpu_mode()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    device = torch.device("cpu")

    structured = LearnedBinaryOperator(latent_dim=8, structured=True).to(device)
    unconstrained = LearnedBinaryOperator(latent_dim=8, structured=False).to(device)
    transformer = TinyTransformer(max_len=test_max_len).to(device)

    opt_s = torch.optim.AdamW(structured.parameters(), lr=2.5e-3, weight_decay=1e-4)
    opt_u = torch.optim.AdamW(unconstrained.parameters(), lr=2.5e-3, weight_decay=1e-4)
    opt_t = torch.optim.AdamW(transformer.parameters(), lr=2e-3, weight_decay=1e-4)

    ce = nn.CrossEntropyLoss()

    for step in range(1, steps + 1):
        tokens, lengths, targets = make_sequence_batch(
            batch_size,
            1,
            train_max_len,
            test_max_len,
            device,
            rng,
        )

        # Structured learned operator
        opt_s.zero_grad(set_to_none=True)
        s_logits = structured(tokens, lengths)
        s_task = ce(s_logits, targets)
        reg = structured.structural_losses()
        s_loss = (
            s_task
            + 0.55 * reg["identity"]
            + 0.55 * reg["inverse"]
            + 0.80 * reg["associativity"]
            + 0.30 * reg["closure"]
            + 0.20 * reg["separation"]
        )
        s_loss.backward()
        torch.nn.utils.clip_grad_norm_(structured.parameters(), 1.0)
        opt_s.step()

        # Same learned operator architecture, but no structural regularizers.
        opt_u.zero_grad(set_to_none=True)
        u_logits = unconstrained(tokens, lengths)
        u_loss = ce(u_logits, targets)
        u_loss.backward()
        torch.nn.utils.clip_grad_norm_(unconstrained.parameters(), 1.0)
        opt_u.step()

        # Transformer baseline.
        opt_t.zero_grad(set_to_none=True)
        t_logits = transformer(tokens)
        t_loss = ce(t_logits, targets)
        t_loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        opt_t.step()

        if progress_callback and (step == 1 or step % 25 == 0 or step == steps):
            progress_callback(
                step,
                steps,
                float(s_task.detach().cpu()),
                float(s_loss.detach().cpu()),
                float(u_loss.detach().cpu()),
                float(t_loss.detach().cpu()),
                {k: float(v.detach().cpu()) for k, v in reg.items()},
            )

    return structured, unconstrained, transformer


@torch.no_grad()
def evaluate_depth(
    structured,
    unconstrained,
    transformer,
    *,
    max_len: int,
    examples_per_length: int,
    seed: int,
    progress_callback=None,
):
    device = torch.device("cpu")
    rng = random.Random(seed)
    structured.eval()
    unconstrained.eval()
    transformer.eval()

    out = {
        "length": [],
        "exact": [],
        "structured": [],
        "unconstrained": [],
        "transformer": [],
    }

    for L in range(1, max_len + 1):
        tokens, lengths, targets = make_sequence_batch(
            examples_per_length, L, L, max_len, device, rng
        )

        exact = targets
        s_pred = structured(tokens, lengths).argmax(dim=1)
        u_pred = unconstrained(tokens, lengths).argmax(dim=1)
        t_pred = transformer(tokens).argmax(dim=1)

        out["length"].append(L)
        out["exact"].append(1.0)
        out["structured"].append(float((s_pred == exact).float().mean()))
        out["unconstrained"].append(float((u_pred == exact).float().mean()))
        out["transformer"].append(float((t_pred == exact).float().mean()))

        if progress_callback and (L == 1 or L % 4 == 0 or L == max_len):
            progress_callback(L, max_len)

    return out


@torch.no_grad()
def learned_table_report(model: LearnedBinaryOperator) -> dict:
    device = next(model.parameters()).device
    proto = model.prototypes()
    table = []
    correct = 0

    for a in range(4):
        row = []
        for b in range(4):
            z = model.op(proto[a:a+1], proto[b:b+1])
            pred = int(model.classify_latent(z).argmax(dim=1).item())
            truth = (a + b) % 4
            correct += int(pred == truth)
            row.append(pred)
        table.append(row)

    # Class-level associativity.
    assoc_ok = 0
    assoc_total = 0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                za = proto[a:a+1]
                zb = proto[b:b+1]
                zc = proto[c:c+1]
                lhs = model.op(model.op(za, zb), zc)
                rhs = model.op(za, model.op(zb, zc))
                lhs_cls = int(model.classify_latent(lhs).argmax(dim=1).item())
                rhs_cls = int(model.classify_latent(rhs).argmax(dim=1).item())
                assoc_ok += int(lhs_cls == rhs_cls)
                assoc_total += 1

    regs = model.structural_losses()

    return {
        "predicted_cayley_table": table,
        "ground_truth_hidden_table": [
            [(a + b) % 4 for b in range(4)] for a in range(4)
        ],
        "table_accuracy": correct / 16.0,
        "class_associativity_accuracy": assoc_ok / assoc_total,
        "structural_residuals": {
            k: float(v.detach().cpu()) for k, v in regs.items()
        },
        "prototype_vectors": model.prototypes().detach().cpu().tolist(),
    }


# -----------------------------
# Topology evaluation
# -----------------------------

@dataclass
class DynamicState:
    name: str
    node_count: int
    edges: List[Tuple[int, int, int]]
    src: int
    dst: int
    target_relation: int
    coherent: bool


def build_adj(node_count: int, edges: List[Tuple[int, int, int]]):
    adj = [[] for _ in range(node_count)]
    for idx, (u, v, r) in enumerate(edges):
        adj[u].append((v, r, idx))
        adj[v].append((u, (-r) % 4, idx))
    return adj


def path_relations(
    node_count: int,
    edges: List[Tuple[int, int, int]],
    src: int,
    dst: int,
) -> Optional[List[int]]:
    if src == dst:
        return []
    adj = build_adj(node_count, edges)
    q = deque([src])
    parent = {src: None}
    parent_rel = {}
    while q:
        u = q.popleft()
        for v, r, _ in adj[u]:
            if v not in parent:
                parent[v] = u
                parent_rel[v] = r
                q.append(v)
                if v == dst:
                    q.clear()
                    break
    if dst not in parent:
        return None

    rels = []
    cur = dst
    while cur != src:
        rels.append(parent_rel[cur])
        cur = parent[cur]
    rels.reverse()
    return rels


def farthest_pair(node_count: int, edges: List[Tuple[int, int, int]]):
    adj = [[] for _ in range(node_count)]
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)

    best = (0, 0, -1)
    for src in range(node_count):
        dist = [-1] * node_count
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for dst, d in enumerate(dist):
            if d > best[2]:
                best = (src, dst, d)
    return best


def make_dynamic_episode(rng: random.Random, node_count: int) -> List[DynamicState]:
    potentials = [rng.randrange(4) for _ in range(node_count)]

    edges = []
    for v in range(1, node_count):
        u = rng.randrange(v)
        r = (potentials[v] - potentials[u]) % 4
        edges.append((u, v, r))

    src, dst, _ = farthest_pair(node_count, edges)
    target = (potentials[dst] - potentials[src]) % 4

    states = [
        DynamicState("BASE", node_count, list(edges), src, dst, target, True)
    ]

    # Find src-dst path edge indices in tree.
    adj = build_adj(node_count, edges)
    q = deque([src])
    parent = {src: None}
    parent_edge = {}
    while q:
        u = q.popleft()
        for v, _, idx in adj[u]:
            if v not in parent:
                parent[v] = u
                parent_edge[v] = idx
                q.append(v)

    path_indices = []
    cur = dst
    while cur != src:
        path_indices.append(parent_edge[cur])
        cur = parent[cur]
    path_indices.reverse()

    cut_idx = path_indices[len(path_indices) // 2]
    edges.pop(cut_idx)
    states.append(
        DynamicState("CUT", node_count, list(edges), src, dst, UNKNOWN, True)
    )

    # Reconnect components with a different correct edge.
    adj2 = build_adj(node_count, edges)
    comp_a = {src}
    q = deque([src])
    while q:
        u = q.popleft()
        for v, _, _ in adj2[u]:
            if v not in comp_a:
                comp_a.add(v)
                q.append(v)
    comp_b = [x for x in range(node_count) if x not in comp_a]

    ua = rng.choice(list(comp_a))
    vb = rng.choice(comp_b)
    reconnect_r = (potentials[vb] - potentials[ua]) % 4
    edges.append((ua, vb, reconnect_r))
    states.append(
        DynamicState("RECONNECT", node_count, list(edges), src, dst, target, True)
    )

    existing = {(min(u, v), max(u, v)) for u, v, _ in edges}
    candidates = [
        (u, v)
        for u in range(node_count)
        for v in range(u + 1, node_count)
        if (u, v) not in existing
    ]
    cu, cv = rng.choice(candidates)
    correct = (potentials[cv] - potentials[cu]) % 4
    wrong = (correct + rng.choice([1, 2, 3])) % 4
    edges.append((cu, cv, wrong))
    states.append(
        DynamicState("CONTRADICTION", node_count, list(edges), src, dst, target, False)
    )

    edges.pop()
    states.append(
        DynamicState("REPAIR", node_count, list(edges), src, dst, target, True)
    )

    return states


@torch.no_grad()
def model_compose_relation_list(
    model: LearnedBinaryOperator,
    rels: List[int],
) -> int:
    if len(rels) == 0:
        return 0
    device = next(model.parameters()).device
    tokens = torch.full((1, len(rels)), UNKNOWN, dtype=torch.long, device=device)
    tokens[0, :] = torch.tensor(rels, dtype=torch.long, device=device)
    lengths = torch.tensor([len(rels)], dtype=torch.long, device=device)
    return int(model(tokens, lengths).argmax(dim=1).item())


def fundamental_cycles(node_count: int, edges: List[Tuple[int, int, int]]):
    """
    Build a spanning forest. Every non-tree edge yields one fundamental cycle.
    Return (tree_path_relations u->v, direct_edge_relation u->v).
    """
    parent = list(range(node_count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    tree_edges = []
    back_edges = []
    for edge in edges:
        u, v, r = edge
        if union(u, v):
            tree_edges.append(edge)
        else:
            back_edges.append(edge)

    cycles = []
    for u, v, r in back_edges:
        rels = path_relations(node_count, tree_edges, u, v)
        if rels is None:
            continue
        cycles.append((rels, r))
    return cycles


@torch.no_grad()
def model_detect_coherence(model: LearnedBinaryOperator, state: DynamicState) -> bool:
    cycles = fundamental_cycles(state.node_count, state.edges)
    for path_rels, direct_r in cycles:
        pred = model_compose_relation_list(model, path_rels)
        if pred != direct_r:
            return False
    return True


def exact_detect_coherence(state: DynamicState) -> bool:
    cycles = fundamental_cycles(state.node_count, state.edges)
    for path_rels, direct_r in cycles:
        if ground_truth_compose(path_rels) != direct_r:
            return False
    return True


def exact_infer_state(state: DynamicState) -> int:
    rels = path_relations(state.node_count, state.edges, state.src, state.dst)
    if rels is None:
        return UNKNOWN
    return ground_truth_compose(rels)


@torch.no_grad()
def model_infer_state(model: LearnedBinaryOperator, state: DynamicState) -> int:
    rels = path_relations(state.node_count, state.edges, state.src, state.dst)
    if rels is None:
        return UNKNOWN
    return model_compose_relation_list(model, rels)
