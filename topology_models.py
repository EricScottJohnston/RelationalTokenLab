from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


UNKNOWN = 4
RELATION_NAMES = ["0 / identity", "+90 / i", "180 / -1", "-90 / -i", "UNKNOWN / disconnected"]


def set_cpu_mode() -> None:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


@dataclass
class GraphSample:
    node_count: int
    edges: List[Tuple[int, int, int]]
    src: int
    dst: int
    relation_target: int
    coherent: bool
    hidden_potentials: List[int]


@dataclass
class GraphBatch:
    node_count: torch.Tensor
    edge_u: torch.Tensor
    edge_v: torch.Tensor
    edge_r: torch.Tensor
    edge_mask: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    relation_target: torch.Tensor
    coherent_target: torch.Tensor


def build_adjacency(node_count: int, edges: List[Tuple[int, int, int]]):
    adj = [[] for _ in range(node_count)]
    for u, v, r in edges:
        r = int(r) % 4
        adj[u].append((v, r))
        adj[v].append((u, (-r) % 4))
    return adj


def infer_relation_exact(
    node_count: int,
    edges: List[Tuple[int, int, int]],
    src: int,
    dst: int,
) -> int:
    if src == dst:
        return 0
    adj = build_adjacency(node_count, edges)
    q = deque([(src, 0)])
    seen = {src}
    while q:
        u, acc = q.popleft()
        for v, rel in adj[u]:
            if v in seen:
                continue
            new_acc = (acc + rel) % 4
            if v == dst:
                return new_acc
            seen.add(v)
            q.append((v, new_acc))
    return UNKNOWN


def check_coherence_exact(
    node_count: int,
    edges: List[Tuple[int, int, int]],
) -> bool:
    adj = build_adjacency(node_count, edges)
    potential: Dict[int, int] = {}
    for root in range(node_count):
        if root in potential:
            continue
        potential[root] = 0
        q = deque([root])
        while q:
            u = q.popleft()
            for v, rel in adj[u]:
                implied = (potential[u] + rel) % 4
                if v not in potential:
                    potential[v] = implied
                    q.append(v)
                elif potential[v] != implied:
                    return False
    return True


def shortest_path_edges(
    node_count: int,
    edges: List[Tuple[int, int, int]],
    src: int,
    dst: int,
) -> Optional[List[Tuple[int, int]]]:
    adj = [[] for _ in range(node_count)]
    for idx, (u, v, r) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    q = deque([src])
    parent = {src: None}
    parent_edge = {}
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v, edge_idx in adj[u]:
            if v not in parent:
                parent[v] = u
                parent_edge[v] = edge_idx
                q.append(v)

    if dst not in parent:
        return None

    out = []
    cur = dst
    while cur != src:
        p = parent[cur]
        out.append((parent_edge[cur], cur))
        cur = p
    out.reverse()
    return out


def farthest_pair(node_count: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
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


def generate_sample(
    rng: random.Random,
    min_nodes: int,
    max_nodes: int,
    max_edges: int,
    disconnected_prob: float = 0.18,
    contradiction_prob: float = 0.35,
) -> GraphSample:
    n = rng.randint(min_nodes, max_nodes)
    potentials = [rng.randrange(4) for _ in range(n)]

    make_disconnected = rng.random() < disconnected_prob and n >= 5

    if make_disconnected:
        cut = rng.randint(2, n - 2)
        components = [list(range(0, cut)), list(range(cut, n))]
    else:
        components = [list(range(n))]

    edges: List[Tuple[int, int, int]] = []

    # Connected tree inside each component.
    for comp in components:
        for idx in range(1, len(comp)):
            v = comp[idx]
            u = comp[rng.randrange(idx)]
            r = (potentials[v] - potentials[u]) % 4
            edges.append((u, v, r))

    # Extra coherent edges inside components.
    desired = min(max_edges, max(len(edges), n + rng.randint(0, max(1, n // 2))))
    attempts = 0
    existing = {(min(u, v), max(u, v)) for u, v, _ in edges}
    while len(edges) < desired and attempts < 1000:
        comp = rng.choice(components)
        if len(comp) < 2:
            attempts += 1
            continue
        u, v = rng.sample(comp, 2)
        key = (min(u, v), max(u, v))
        if key not in existing:
            r = (potentials[v] - potentials[u]) % 4
            edges.append((u, v, r))
            existing.add(key)
        attempts += 1

    if make_disconnected:
        src = rng.choice(components[0])
        dst = rng.choice(components[1])
        relation_target = UNKNOWN
    else:
        src, dst = rng.sample(range(n), 2)
        relation_target = (potentials[dst] - potentials[src]) % 4

    coherent = True
    if rng.random() < contradiction_prob and edges:
        idx = rng.randrange(len(edges))
        u, v, r = edges[idx]
        wrong = (r + rng.choice([1, 2, 3])) % 4
        edges[idx] = (u, v, wrong)
        coherent = False

    return GraphSample(
        node_count=n,
        edges=edges,
        src=src,
        dst=dst,
        relation_target=relation_target,
        coherent=coherent,
        hidden_potentials=potentials,
    )


def collate_samples(
    samples: List[GraphSample],
    max_nodes: int,
    max_edges: int,
    device: torch.device,
) -> GraphBatch:
    B = len(samples)
    edge_u = torch.zeros((B, max_edges), dtype=torch.long, device=device)
    edge_v = torch.zeros((B, max_edges), dtype=torch.long, device=device)
    edge_r = torch.zeros((B, max_edges), dtype=torch.long, device=device)
    edge_mask = torch.zeros((B, max_edges), dtype=torch.bool, device=device)

    node_count = torch.zeros(B, dtype=torch.long, device=device)
    src = torch.zeros(B, dtype=torch.long, device=device)
    dst = torch.zeros(B, dtype=torch.long, device=device)
    relation_target = torch.zeros(B, dtype=torch.long, device=device)
    coherent_target = torch.zeros(B, dtype=torch.long, device=device)

    for b, s in enumerate(samples):
        node_count[b] = s.node_count
        src[b] = s.src
        dst[b] = s.dst
        relation_target[b] = s.relation_target
        coherent_target[b] = 1 if s.coherent else 0
        for e, (u, v, r) in enumerate(s.edges[:max_edges]):
            edge_u[b, e] = u
            edge_v[b, e] = v
            edge_r[b, e] = r
            edge_mask[b, e] = True

    return GraphBatch(
        node_count=node_count,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_r=edge_r,
        edge_mask=edge_mask,
        src=src,
        dst=dst,
        relation_target=relation_target,
        coherent_target=coherent_target,
    )


def node_fourier(ids: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """
    Fixed node-ID encoding so unseen node numbers are defined at test time.
    This deliberately avoids a learned embedding table tied to train graph size.
    """
    ids = ids.float().unsqueeze(-1)
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=ids.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half - 1))
    )
    angles = ids * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TopologyTransformer(nn.Module):
    """
    Generic learned baseline over graph edge records.

    Each edge record contains:
      fixed Fourier node encoding(u),
      fixed Fourier node encoding(v),
      learned relation embedding(r).

    A query token contains source and target node encodings.

    The transformer must learn graph inference and coherence statistically.
    """

    def __init__(
        self,
        max_edges: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        ff: int = 128,
        node_fourier_dim: int = 16,
    ):
        super().__init__()
        self.max_edges = max_edges
        self.node_fourier_dim = node_fourier_dim
        self.rel_emb = nn.Embedding(4, 16)
        self.edge_proj = nn.Linear(node_fourier_dim * 2 + 16, d_model)
        self.query_proj = nn.Linear(node_fourier_dim * 2, d_model)

        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, max_edges + 2, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.rel_head = nn.Linear(d_model, 5)
        self.coh_head = nn.Linear(d_model, 2)

    def forward(self, batch: GraphBatch):
        ufeat = node_fourier(batch.edge_u, self.node_fourier_dim)
        vfeat = node_fourier(batch.edge_v, self.node_fourier_dim)
        rfeat = self.rel_emb(batch.edge_r)
        edges = self.edge_proj(torch.cat([ufeat, vfeat, rfeat], dim=-1))

        qfeat = torch.cat(
            [
                node_fourier(batch.src, self.node_fourier_dim),
                node_fourier(batch.dst, self.node_fourier_dim),
            ],
            dim=-1,
        )
        qtok = self.query_proj(qfeat).unsqueeze(1)

        B = edges.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, qtok, edges], dim=1)
        x = x + self.pos[:, : x.size(1)]

        pad = torch.zeros((B, self.max_edges + 2), dtype=torch.bool, device=x.device)
        pad[:, 2:] = ~batch.edge_mask

        h = self.encoder(x, src_key_padding_mask=pad)
        rel_logits = self.rel_head(h[:, 1])
        coh_logits = self.coh_head(h[:, 0])
        return rel_logits, coh_logits


class MessagePassingGNN(nn.Module):
    """
    A real-valued message-passing baseline with a fixed number of rounds.

    It is graph-native but does NOT receive the Z4 composition rule.
    Fixed message depth makes the train-vs-test reasoning-depth question explicit.
    """

    def __init__(
        self,
        max_nodes: int,
        hidden: int = 64,
        rounds: int = 4,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden = hidden
        self.rounds = rounds

        # Initial node features: source marker, target marker, bias.
        self.node_init = nn.Linear(3, hidden)
        self.rel_emb = nn.Embedding(4, 16)

        self.msg = nn.Sequential(
            nn.Linear(hidden + 16, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.update = nn.GRUCell(hidden, hidden)

        self.rel_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 5),
        )
        self.coh_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, batch: GraphBatch):
        B = batch.edge_u.size(0)
        N = self.max_nodes
        device = batch.edge_u.device

        node_feat = torch.zeros((B, N, 3), dtype=torch.float32, device=device)
        node_feat[..., 2] = 1.0
        bidx = torch.arange(B, device=device)
        node_feat[bidx, batch.src, 0] = 1.0
        node_feat[bidx, batch.dst, 1] = 1.0
        h = self.node_init(node_feat)

        for _ in range(self.rounds):
            agg = torch.zeros_like(h)
            for e in range(batch.edge_u.size(1)):
                mask = batch.edge_mask[:, e]
                if not mask.any():
                    continue
                u = batch.edge_u[:, e]
                v = batch.edge_v[:, e]
                r = batch.edge_r[:, e]

                hu = h[bidx, u]
                hv = h[bidx, v]
                er = self.rel_emb(r)
                inv_er = self.rel_emb((-r) % 4)

                msg_uv = self.msg(torch.cat([hu, er], dim=-1))
                msg_vu = self.msg(torch.cat([hv, inv_er], dim=-1))

                msg_uv = msg_uv * mask.unsqueeze(-1)
                msg_vu = msg_vu * mask.unsqueeze(-1)

                agg[bidx, v] += msg_uv
                agg[bidx, u] += msg_vu

            h = self.update(
                agg.reshape(B * N, self.hidden),
                h.reshape(B * N, self.hidden),
            ).reshape(B, N, self.hidden)

        hs = h[bidx, batch.src]
        hd = h[bidx, batch.dst]
        rel_logits = self.rel_head(torch.cat([hs, hd], dim=-1))

        # Pool only active nodes.
        node_ids = torch.arange(N, device=device).unsqueeze(0)
        active = node_ids < batch.node_count.unsqueeze(1)
        masked = h * active.unsqueeze(-1)
        mean_pool = masked.sum(dim=1) / active.sum(dim=1, keepdim=True).clamp(min=1)
        max_pool = h.masked_fill(~active.unsqueeze(-1), -1e9).max(dim=1).values
        coh_logits = self.coh_head(torch.cat([mean_pool, max_pool], dim=-1))

        return rel_logits, coh_logits


def make_training_batch(
    *,
    rng: random.Random,
    batch_size: int,
    train_min_nodes: int,
    train_max_nodes: int,
    max_nodes: int,
    max_edges: int,
    device: torch.device,
) -> GraphBatch:
    samples = [
        generate_sample(
            rng,
            min_nodes=train_min_nodes,
            max_nodes=train_max_nodes,
            max_edges=min(max_edges, train_max_nodes + 4),
        )
        for _ in range(batch_size)
    ]
    return collate_samples(samples, max_nodes, max_edges, device)


def train_baselines(
    *,
    seed: int = 19,
    train_min_nodes: int = 4,
    train_max_nodes: int = 8,
    test_max_nodes: int = 24,
    max_edges: int = 40,
    steps: int = 850,
    batch_size: int = 96,
    progress_callback=None,
):
    set_cpu_mode()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    device = torch.device("cpu")

    transformer = TopologyTransformer(max_edges=max_edges).to(device)
    gnn = MessagePassingGNN(max_nodes=test_max_nodes, rounds=4).to(device)

    opt_t = torch.optim.AdamW(transformer.parameters(), lr=2e-3, weight_decay=1e-4)
    opt_g = torch.optim.AdamW(gnn.parameters(), lr=2e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction="none")

    history = {"transformer": [], "gnn": []}

    for step in range(1, steps + 1):
        batch = make_training_batch(
            rng=rng,
            batch_size=batch_size,
            train_min_nodes=train_min_nodes,
            train_max_nodes=train_max_nodes,
            max_nodes=test_max_nodes,
            max_edges=max_edges,
            device=device,
        )

        # Query relation is only well-defined from the graph if coherent.
        rel_mask = batch.coherent_target.bool()

        opt_t.zero_grad(set_to_none=True)
        t_rel, t_coh = transformer(batch)
        t_rel_loss = ce(t_rel, batch.relation_target)
        if rel_mask.any():
            t_rel_loss = t_rel_loss[rel_mask].mean()
        else:
            t_rel_loss = t_rel_loss.mean() * 0.0
        t_coh_loss = ce(t_coh, batch.coherent_target).mean()
        t_loss = t_rel_loss + t_coh_loss
        t_loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        opt_t.step()

        opt_g.zero_grad(set_to_none=True)
        g_rel, g_coh = gnn(batch)
        g_rel_loss = ce(g_rel, batch.relation_target)
        if rel_mask.any():
            g_rel_loss = g_rel_loss[rel_mask].mean()
        else:
            g_rel_loss = g_rel_loss.mean() * 0.0
        g_coh_loss = ce(g_coh, batch.coherent_target).mean()
        g_loss = g_rel_loss + g_coh_loss
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
        opt_g.step()

        history["transformer"].append(float(t_loss.detach().cpu()))
        history["gnn"].append(float(g_loss.detach().cpu()))

        if progress_callback and (step == 1 or step % 25 == 0 or step == steps):
            progress_callback(
                step,
                steps,
                float(t_loss.detach().cpu()),
                float(g_loss.detach().cpu()),
            )

    return transformer, gnn, history


def make_dynamic_episode(
    rng: random.Random,
    node_count: int,
) -> List[Tuple[str, GraphSample]]:
    """
    Dynamic sequence:

      BASE:
        coherent tree, source and target chosen far apart.

      CUT:
        remove one edge on their unique path -> source/target disconnected.

      RECONNECT:
        add a correct cross-component relation -> relation restored.

      CONTRADICTION:
        add a wrong edge -> coherence defect.

      REPAIR:
        remove the wrong edge -> coherence restored.

    No learning/retraining occurs between states.
    """
    potentials = [rng.randrange(4) for _ in range(node_count)]

    # Random tree.
    edges: List[Tuple[int, int, int]] = []
    for v in range(1, node_count):
        u = rng.randrange(v)
        r = (potentials[v] - potentials[u]) % 4
        edges.append((u, v, r))

    src, dst, distance = farthest_pair(node_count, edges)
    relation = (potentials[dst] - potentials[src]) % 4

    states: List[Tuple[str, GraphSample]] = []

    def sample_from(current_edges, rel_target, coherent):
        return GraphSample(
            node_count=node_count,
            edges=list(current_edges),
            src=src,
            dst=dst,
            relation_target=rel_target,
            coherent=coherent,
            hidden_potentials=potentials,
        )

    states.append(("BASE", sample_from(edges, relation, True)))

    # Remove a central edge on unique src-dst tree path.
    path = shortest_path_edges(node_count, edges, src, dst)
    edge_idx = path[len(path) // 2][0]
    removed = edges.pop(edge_idx)
    states.append(("CUT", sample_from(edges, UNKNOWN, True)))

    # Identify components and reconnect with a correct edge.
    adj = [[] for _ in range(node_count)]
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)
    comp_a = set()
    q = deque([src])
    comp_a.add(src)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in comp_a:
                comp_a.add(v)
                q.append(v)
    comp_b = [v for v in range(node_count) if v not in comp_a]
    ua = rng.choice(list(comp_a))
    vb = rng.choice(comp_b)
    correct = (potentials[vb] - potentials[ua]) % 4
    reconnect_edge = (ua, vb, correct)
    edges.append(reconnect_edge)
    states.append(("RECONNECT", sample_from(edges, relation, True)))

    # Add a contradictory edge not already present.
    existing = {(min(u, v), max(u, v)) for u, v, _ in edges}
    candidates = [
        (u, v)
        for u in range(node_count)
        for v in range(u + 1, node_count)
        if (u, v) not in existing
    ]
    cu, cv = rng.choice(candidates)
    correct_cv = (potentials[cv] - potentials[cu]) % 4
    wrong = (correct_cv + rng.choice([1, 2, 3])) % 4
    bad_edge = (cu, cv, wrong)
    edges.append(bad_edge)
    states.append(("CONTRADICTION", sample_from(edges, relation, False)))

    # Repair.
    edges.pop()
    states.append(("REPAIR", sample_from(edges, relation, True)))

    return states
