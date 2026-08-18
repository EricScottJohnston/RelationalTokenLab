"""Experiment 7 — lexicon bootstrapping.  Corpus generation.

THE IDEA
    Experiment 6 showed the reader works when it recognises a relation word and
    fails when it does not, and that composition on top of a correct reading is
    essentially exact (0.998). So: use the composition as an equation and solve
    it for the words you do not know.

        kleen  --raises-->  spunt  --gorbles-->  frell
        kleen  --reduces-->                      frell

        positive XOR gorbles = negative   ->   gorbles is negative

    This module builds the corpus that test rests on.

WHY THE CORPUS NEEDS A LATENT POTENTIAL
    The solve only works if the world is consistent -- if every way of getting
    from one entity to another agrees. A graph has that property exactly when
    every cycle XORs to zero, and a graph has *that* property exactly when a
    per-entity potential exists:

        polarity(a -> b) = phi(a) XOR phi(b)

    That is a theorem, not a modelling choice: consistency and the existence of
    a potential are the same condition. So generating from a potential is the
    general way to generate a consistent world, and setting `consistent=False`
    (polarities drawn at random) is the sharpest possible control -- it removes
    the algebra and nothing else.

WHY ENTITIES ARE SHARED ACROSS KINDS
    Each kind gets its own potential over the *same* entity pool. If each kind
    had its own entities, a hidden phrase's kind could be read off its
    arguments and kind inference would be free. Sharing the pool forces kind to
    be decided by which system the phrase's edges are consistent with, which is
    the thing being tested.

WHAT IS EXCLUDED AND WHY
    Only transitive relations appear. A non-transitive relation has no
    composition, so it carries no equation and could never be identified by
    this mechanism; including one would also inject false inconsistencies into
    a graph that is otherwise sound. EVIDENTIAL is dropped whole, since neither
    INDICATES nor CONTRADICTS is transitive.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from relational_lexicon import ALL_RELATIONS, KINDS, Relation
from sentence_data import PLAIN_FRAMES, nonce, render

TRANSITIVE = [r for r in ALL_RELATIONS if r.transitive]
USABLE_KINDS = sorted({r.kind for r in TRANSITIVE})

# relation lookup: kind -> polarity -> relations
BY_KIND_POLARITY: Dict[str, Dict[int, List[Relation]]] = {}
for _r in TRANSITIVE:
    BY_KIND_POLARITY.setdefault(_r.kind, {}).setdefault(_r.polarity, []).append(_r)

# Kinds that offer both polarities. A kind with only one polarity would make
# every edge in it the same sign, so nothing about polarity could be learned.
BALANCED_KINDS = [k for k in USABLE_KINDS if len(BY_KIND_POLARITY[k]) == 2]

ALL_PHRASES: List[Tuple[str, Relation]] = [
    (p, r) for r in TRANSITIVE for p in r.phrases
]
PHRASE_OWNER: Dict[str, Relation] = {p: r for p, r in ALL_PHRASES}


@dataclass
class Edge:
    """One sentence, and the relation it actually expresses."""
    source: str
    target: str
    kind: str
    polarity: int          # ground truth
    relation: str          # ground truth
    phrase: str
    text: str
    hidden: bool           # the reader does not know this phrase
    observed_polarity: int | None = None   # what the reader reported
    corrupted: bool = False                # reader got it wrong


@dataclass
class Corpus:
    edges: List[Edge]
    entities: List[str]
    hidden_phrases: List[str]
    known_phrases: List[str]
    kinds: List[str]
    potentials: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def phrase_truth(self) -> Dict[str, Tuple[str, int]]:
        return {p: (PHRASE_OWNER[p].kind, PHRASE_OWNER[p].polarity)
                for p in self.hidden_phrases}

    def hidden_present(self) -> List[str]:
        """Hidden phrases that actually occur. A phrase with no sentence in the
        corpus is not something the mechanism could ever recover, and counting
        it against the score would just be measuring the sampler."""
        seen = {e.phrase for e in self.edges if e.hidden}
        return [p for p in self.hidden_phrases if p in seen]

    def occurrences(self) -> Dict[str, int]:
        """How many sentences each hidden phrase appears in.

        This is the number that decides whether kind can be settled at all. One
        occurrence tells you nothing: a single equation with one unknown is
        satisfiable in every kind, so every kind survives and the phrase is
        ambiguous. With m occurrences a wrong kind has to agree with itself m
        times by chance, which is 2^-(m-1). Below about four occurrences the
        mechanism cannot be expected to work, and that is a property of the
        corpus, not of the solver.
        """
        counts: Dict[str, int] = {}
        for e in self.edges:
            if e.hidden:
                counts[e.phrase] = counts.get(e.phrase, 0) + 1
        return counts

    def summary(self) -> Dict[str, object]:
        occ = sorted(self.occurrences().values())
        per_kind = {}
        for k in self.kinds:
            es = [e for e in self.edges if e.kind == k]
            per_kind[k] = {
                "edges": len(es),
                "entities": len({e.source for e in es} | {e.target for e in es}),
                "hidden_edges": sum(1 for e in es if e.hidden),
            }
        return {
            "entities": len(self.entities),
            "edges": len(self.edges),
            "kinds": self.kinds,
            "phrases_total": len(ALL_PHRASES),
            "phrases_hidden": len(self.hidden_phrases),
            "phrases_hidden_and_present": len(self.hidden_present()),
            "hidden_edges": sum(1 for e in self.edges if e.hidden),
            "corrupted_edges": sum(1 for e in self.edges if e.corrupted),
            "occurrences_per_hidden_phrase": {
                "min": occ[0] if occ else 0,
                "median": occ[len(occ) // 2] if occ else 0,
                "max": occ[-1] if occ else 0,
                "at_or_below_3": sum(1 for v in occ if v <= 3),
            },
            "per_kind": per_kind,
        }


def build_corpus(
    *,
    seed: int = 91,
    n_entities: int = 60,
    edges_per_kind: int = 900,
    hidden_fraction: float = 0.3,
    reader_error: float = 0.0,
    consistent: bool = True,
    kinds: Sequence[str] | None = None,
) -> Corpus:
    """Build a corpus and hide a fraction of the lexicon from the reader.

    hidden_fraction is over *phrases*, not relations, which is the realistic
    case: the reader may know "prevents" and not know "blocks", and the whole
    point is to recover that "blocks" is the same relation.

    reader_error flips the polarity the reader reports on a known edge. It
    stands in for an imperfect front end and, since a flipped edge contradicts
    the rest of the graph, it is also what the contradiction detector is
    measured against.

    consistent=False draws polarities at random instead of from a potential.
    That is the control: the algebra is removed and nothing else changes.
    """
    rng = random.Random(seed)
    ks = list(kinds) if kinds else list(BALANCED_KINDS)

    entities: List[str] = []
    seen = set()
    while len(entities) < n_entities:
        e = nonce(rng)
        if e not in seen and all(e not in o and o not in e for o in entities):
            seen.add(e)
            entities.append(e)

    # Hide phrases.
    phrases = [p for p, _ in ALL_PHRASES if PHRASE_OWNER[p].kind in ks]
    rng.shuffle(phrases)
    n_hidden = int(round(len(phrases) * hidden_fraction))
    hidden = set(phrases[:n_hidden])
    known = phrases[n_hidden:]

    potentials: Dict[str, Dict[str, int]] = {}
    edges: List[Edge] = []
    for k in ks:
        phi = {e: rng.randint(0, 1) for e in entities}
        potentials[k] = phi
        pol_choices = BY_KIND_POLARITY[k]
        placed = set()
        made = 0
        guard = 0
        while made < edges_per_kind and guard < edges_per_kind * 40:
            guard += 1
            a, b = rng.sample(entities, 2)
            if (a, b) in placed:
                continue
            polarity = (phi[a] ^ phi[b]) if consistent else rng.randint(0, 1)
            opts = pol_choices.get(polarity)
            if not opts:
                continue
            rel = rng.choice(opts)
            phrase = rng.choice(rel.phrases)
            frame = rng.choice(PLAIN_FRAMES)
            r = render(frame, rng, p=phrase)
            # render() supplies its own nonce entities; substitute the corpus
            # ones back in so the graph is over a shared entity pool.
            if r is None:
                continue
            text, ra, rb = r
            text = text.replace(ra, a).replace(rb, b)
            if text.lower().count(a) != 1 or text.lower().count(b) != 1:
                continue
            placed.add((a, b))
            made += 1
            is_hidden = phrase in hidden
            obs = None
            corrupted = False
            if not is_hidden:
                obs = polarity
                if reader_error > 0 and rng.random() < reader_error:
                    obs ^= 1
                    corrupted = True
            edges.append(Edge(source=a, target=b, kind=k, polarity=polarity,
                              relation=rel.name, phrase=phrase, text=text,
                              hidden=is_hidden, observed_polarity=obs,
                              corrupted=corrupted))

    rng.shuffle(edges)
    return Corpus(edges=edges, entities=entities,
                  hidden_phrases=sorted(hidden), known_phrases=sorted(known),
                  kinds=ks, potentials=potentials)


def verify_consistency(corpus: Corpus) -> Dict[str, object]:
    """Ground-truth check: every cycle in every kind must XOR to zero.

    Tested against the potential directly, which is what the solver has to
    rediscover without being told it exists.
    """
    out: Dict[str, object] = {}
    for k in corpus.kinds:
        phi = corpus.potentials[k]
        bad = [e for e in corpus.edges
               if e.kind == k and (phi[e.source] ^ phi[e.target]) != e.polarity]
        out[k] = {"edges_violating_potential": len(bad)}
    return out


if __name__ == "__main__":
    import json
    c = build_corpus()
    print(json.dumps(c.summary(), indent=2))
    print(json.dumps(verify_consistency(c), indent=2))
    print("\nsample sentences:")
    for e in c.edges[:8]:
        flag = "HIDDEN " if e.hidden else "known  "
        print(f"  {flag}{e.text}   [{e.source} -> {e.target}] "
              f"{e.relation}/{e.kind} pol={e.polarity} phrase='{e.phrase}'")
