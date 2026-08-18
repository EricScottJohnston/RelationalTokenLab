"""Experiment 7 — the solver.

Polarity is a Z2 group, so a chain of relations composes by XOR. Turn that
round and it becomes a linear system over GF(2).

THE SYSTEM
    Give every entity an unknown potential psi(e) inside each kind. Every
    sentence the reader understands is then one equation:

        psi(a) XOR psi(b) = polarity

    and every sentence containing a phrase the reader does *not* know is an
    equation with one extra unknown:

        psi(a) XOR psi(b) XOR x_p = 0

    Solve. psi is only determined up to a constant per connected component --
    the gauge is free -- but x_p is a difference of potentials and so is
    gauge-invariant, which is why the unknown phrase can come out determined
    even though the potentials never do.

KIND
    Kind is not a group and does not compose; it is a constraint. Composition
    is defined only *within* a kind, so a phrase's edges must be consistent
    with the system of the kind it belongs to and inconsistent with the others.
    That gives a hypothesis test: try the phrase in all four systems, keep the
    kind that survives. A phrase surviving in more than one system is
    ambiguous and is not claimed.

    This is why the mechanism can recover what Experiment 6's reader could not.
    The reader saw kind collapse on unseen wording (0.35-0.57) because a single
    sentence carries no signal for it. Consistency across many chains does.

IDENTIFIABILITY IS A REAL ANSWER
    After elimination a variable is determined only if its row involves no free
    variables. A phrase the corpus does not pin down comes back unidentified
    rather than guessed. That is the same property the lexicon has when two
    relations cross kinds: an absence of a defined product, not a low
    confidence.

CONTRADICTION
    Rows are inserted one at a time. A row that reduces to 0 = 1 contradicts
    what is already in the system; it is recorded and skipped, so the system
    stays solvable and the conflicts are reported. That is contradiction
    detection and robustness from one mechanism, with nothing judging anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from bootstrap_data import Corpus, Edge, PHRASE_OWNER


class GF2:
    """Incremental Gaussian elimination over GF(2).

    A row is one integer: bits 0..n-1 are variable coefficients, bit n is the
    right-hand side.
    """

    __slots__ = ("n", "mask", "pivots")

    def __init__(self, n_vars: int, pivots: Dict[int, int] | None = None):
        self.n = n_vars
        self.mask = (1 << n_vars) - 1
        self.pivots: Dict[int, int] = dict(pivots) if pivots else {}

    def copy(self) -> "GF2":
        return GF2(self.n, self.pivots)

    def _reduce(self, row: int) -> int:
        cur = row
        while True:
            v = cur & self.mask
            if v == 0:
                return cur
            c = v.bit_length() - 1
            p = self.pivots.get(c)
            if p is None:
                return cur
            cur ^= p

    def add(self, row: int) -> bool:
        """Insert a row. False means it contradicts the rows already present."""
        cur = self._reduce(row)
        v = cur & self.mask
        if v == 0:
            return not ((cur >> self.n) & 1)
        self.pivots[v.bit_length() - 1] = cur
        return True

    def rref(self) -> None:
        """Reduce every pivot row against the others.

        Descending order is what makes one pass enough: clearing column c from
        a row can only introduce bits below c, and those columns have not been
        processed yet.
        """
        for c in sorted(self.pivots, reverse=True):
            r = self.pivots[c]
            for c2, r2 in self.pivots.items():
                if c2 != c and (r2 >> c) & 1:
                    self.pivots[c2] = r2 ^ r

    def value(self, var: int) -> int | None:
        """The variable's value if the system determines it, else None."""
        row = self.pivots.get(var)
        if row is None:
            return None                       # free variable
        if (row & self.mask) ^ (1 << var):
            return None                       # row still involves free variables
        return (row >> self.n) & 1


@dataclass
class Identification:
    phrase: str
    kind: str
    polarity: int
    round: int


@dataclass
class BootstrapResult:
    identified: Dict[str, Tuple[str, int]]
    rounds: List[Dict[str, object]]
    conflicts: Dict[str, List[int]] = field(default_factory=dict)
    unidentified: List[str] = field(default_factory=list)


def _build_base(corpus: Corpus, kind: str, ent_index: Dict[str, int],
                learned: Dict[str, Tuple[str, int]]) -> Tuple[GF2, List[int]]:
    """The system for one kind, from everything currently known."""
    sys = GF2(len(ent_index) + 1)          # last slot reserved for a candidate
    conflicts: List[int] = []
    for i, e in enumerate(corpus.edges):
        pol = None
        if not e.hidden:
            if e.kind == kind:
                pol = e.observed_polarity
        else:
            got = learned.get(e.phrase)
            if got is not None and got[0] == kind:
                pol = got[1]
        if pol is None:
            continue
        row = ((1 << ent_index[e.source]) | (1 << ent_index[e.target])
               | (pol << sys.n))
        if not sys.add(row):
            conflicts.append(i)
    sys.rref()
    return sys, conflicts


def _test(base: GF2, edges: Sequence[Edge], ent_index: Dict[str, int]) -> Tuple[str, int | None]:
    """Try an unknown phrase inside one kind's system.

    The solver does not know what kind the phrase's sentences belong to, so
    every sentence containing it is offered to every candidate system.
    """
    sys = base.copy()
    cand = len(ent_index)
    for e in edges:
        row = ((1 << ent_index[e.source]) | (1 << ent_index[e.target])
               | (1 << cand))
        if not sys.add(row):
            return ("inconsistent", None)
    sys.rref()
    v = sys.value(cand)
    return ("determined", v) if v is not None else ("consistent", None)


def bootstrap(corpus: Corpus, max_rounds: int = 12) -> BootstrapResult:
    """Identify unknown phrases, then use them to identify more.

    A phrase is claimed only when exactly one kind is consistent with it and
    that kind pins its polarity down. Everything else is left unidentified.
    """
    ent_index = {e: i for i, e in enumerate(corpus.entities)}
    edges_by_phrase: Dict[str, List[Edge]] = {}
    for e in corpus.edges:
        if e.hidden:
            edges_by_phrase.setdefault(e.phrase, []).append(e)

    learned: Dict[str, Tuple[str, int]] = {}
    pending = set(edges_by_phrase)
    rounds: List[Dict[str, object]] = []
    conflicts: Dict[str, List[int]] = {}

    for rnd in range(1, max_rounds + 1):
        bases = {}
        for k in corpus.kinds:
            bases[k], conflicts[k] = _build_base(corpus, k, ent_index, learned)

        newly: Dict[str, Tuple[str, int]] = {}
        for p in sorted(pending):
            edges = edges_by_phrase[p]
            survivors = []
            for k in corpus.kinds:
                status, val = _test(bases[k], edges, ent_index)
                if status != "inconsistent":
                    survivors.append((k, status, val))
            if len(survivors) == 1 and survivors[0][1] == "determined":
                newly[p] = (survivors[0][0], survivors[0][2])

        rounds.append({
            "round": rnd,
            "identified_this_round": len(newly),
            "identified_total": len(learned) + len(newly),
            "still_unknown": len(pending) - len(newly),
        })
        if not newly:
            break
        learned.update(newly)
        pending -= set(newly)

    return BootstrapResult(identified=learned, rounds=rounds,
                           conflicts=conflicts, unidentified=sorted(pending))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score(corpus: Corpus, result: BootstrapResult) -> Dict[str, object]:
    present = corpus.hidden_present()
    truth = corpus.phrase_truth()
    identified = {p: v for p, v in result.identified.items() if p in truth}
    correct = sum(1 for p, v in identified.items() if v == truth[p])
    n = max(len(present), 1)
    return {
        "hidden_present": len(present),
        "identified": len(identified),
        "identified_fraction": len(identified) / n,
        "correct": correct,
        "overall": correct / n,
        "accuracy_among_identified": correct / max(len(identified), 1),
        "kind_correct": sum(1 for p, v in identified.items() if v[0] == truth[p][0]) / n,
        "polarity_correct": sum(1 for p, v in identified.items() if v[1] == truth[p][1]) / n,
        "rounds_used": len(result.rounds),
        "unidentified": len(present) - len(identified),
    }


def baseline_random(corpus: Corpus, seed: int) -> Dict[str, float]:
    import random
    rng = random.Random(seed)
    truth = corpus.phrase_truth()
    present = corpus.hidden_present()
    hits = sum(1 for p in present
               if (rng.choice(corpus.kinds), rng.randint(0, 1)) == truth[p])
    return {"overall": hits / max(len(present), 1)}


def baseline_majority(corpus: Corpus) -> Dict[str, float]:
    from collections import Counter
    truth = corpus.phrase_truth()
    present = corpus.hidden_present()
    known = Counter((PHRASE_OWNER[p].kind, PHRASE_OWNER[p].polarity)
                    for p in corpus.known_phrases)
    if not known or not present:
        return {"overall": 0.0}
    top = known.most_common(1)[0][0]
    return {"overall": sum(1 for p in present if truth[p] == top) / len(present)}


def contradiction_report(corpus: Corpus, result: BootstrapResult) -> Dict[str, object]:
    """How many contradictions were found, and were they the injected ones.

    Attribution is greedy -- when two rows disagree the second one in is
    blamed -- so precision below 1.0 does not mean a contradiction was missed.
    The count is the reliable number.
    """
    flagged = sorted({i for ids in result.conflicts.values() for i in ids})
    injected = [i for i, e in enumerate(corpus.edges) if e.corrupted]
    hit = sum(1 for i in flagged if corpus.edges[i].corrupted)
    return {
        "injected": len(injected),
        "flagged": len(flagged),
        "detection_ratio": len(flagged) / max(len(injected), 1),
        "flagged_that_were_corrupt": hit,
        "attribution_precision": hit / max(len(flagged), 1),
    }
