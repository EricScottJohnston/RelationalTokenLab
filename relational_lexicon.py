"""Base relational lexicon — the closed class.

The premise, stated plainly: stems are an open class and operations are a
closed one. You cannot enumerate every noun. You *can* enumerate the ways
English says that one thing stands in a relation to another. There are a few
hundred of them and the set changes over centuries.

So this file is the permanent training set. Everything the model will ever
know about *how relations are expressed* lives here. Everything it knows about
any particular subject arrives at runtime from a retrieved document.

Structure of a relation
-----------------------
Each relation carries two things that compose:

    KIND      what sort of relation it is (causal, structural, temporal,
              comparative, evidential). Kinds do not mix. Composing across
              kinds has no product, and that is where a genuine UNKNOWN comes
              from - not a hedge, an absence of a defined composition.

    POLARITY  a Z2 group. Promoting versus suppressing, holding versus
              failing. Double suppression is promotion: preventing a
              prevention enables. Composition is XOR, exactly as in the
              derivational experiment.

Within a kind, composition along a chain is:

    kind(A,B) . kind(B,C) -> kind(A,C),  polarity = p1 XOR p2

Surface variation is the point. Every relation below has many phrasings, and
several of them share no words with each other. The model has to recover the
relation from the phrase, which is the induction problem; the composition is
then arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# Kinds. Composition is defined inside a kind and undefined across kinds.
# --------------------------------------------------------------------------
KINDS = ["CAUSAL", "STRUCTURAL", "TEMPORAL", "COMPARATIVE", "EVIDENTIAL"]
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}

# Polarity is Z2: 0 promotes/holds, 1 suppresses/fails.
POSITIVE, NEGATIVE = 0, 1


@dataclass(frozen=True)
class Relation:
    name: str
    kind: str
    polarity: int
    transitive: bool
    phrases: Tuple[str, ...]

    def angle_target(self) -> Tuple[int, int]:
        """The (kind, polarity) pair the resolver must recover."""
        return (KIND_INDEX[self.kind], self.polarity)


# --------------------------------------------------------------------------
# CAUSAL. The workhorse. Transitive, polarity-carrying.
# --------------------------------------------------------------------------
CAUSAL: List[Relation] = [
    Relation("CAUSES", "CAUSAL", POSITIVE, True, (
        "causes", "leads to", "results in", "produces", "gives rise to",
        "brings about", "induces", "triggers", "generates", "creates",
        "drives", "is responsible for", "is the cause of", "sets off",
        "precipitates", "provokes", "elicits", "yields", "engenders",
        "culminates in", "issues in", "occasions",
    )),
    Relation("PREVENTS", "CAUSAL", NEGATIVE, True, (
        "prevents", "blocks", "stops", "inhibits", "suppresses", "impedes",
        "obstructs", "precludes", "averts", "forestalls", "arrests",
        "counteracts", "neutralizes", "rules out", "shuts down",
        "puts a stop to", "guards against", "protects against", "wards off",
        "eliminates", "negates", "cancels",
    )),
    Relation("ENABLES", "CAUSAL", POSITIVE, True, (
        "enables", "allows", "permits", "makes possible", "facilitates",
        "supports", "sustains", "admits of", "opens the way for",
        "provides for", "licenses", "affords", "lets", "clears the way for",
        "underwrites", "authorizes",
    )),
    Relation("REQUIRES", "CAUSAL", POSITIVE, True, (
        "requires", "depends on", "relies on", "needs", "presupposes",
        "is contingent on", "is conditional on", "calls for", "demands",
        "is predicated on", "hinges on", "rests on", "is a prerequisite for",
        "cannot occur without", "is necessary for",
    )),
    Relation("INCREASES", "CAUSAL", POSITIVE, True, (
        "increases", "raises", "elevates", "amplifies", "intensifies",
        "boosts", "augments", "heightens", "strengthens", "escalates",
        "magnifies", "drives up", "pushes up", "adds to", "compounds",
        "aggravates", "exacerbates", "worsens",
    )),
    Relation("DECREASES", "CAUSAL", NEGATIVE, True, (
        "decreases", "reduces", "lowers", "diminishes", "attenuates",
        "dampens", "weakens", "mitigates", "alleviates", "curtails",
        "drives down", "cuts", "lessens", "moderates", "tempers",
        "relieves", "eases", "subdues",
    )),
    Relation("FAILS_TO_PRODUCE", "CAUSAL", NEGATIVE, True, (
        "fails to produce", "does not cause", "does not lead to",
        "has no effect on", "leaves unchanged", "does not affect",
        "is unrelated to", "does not influence", "makes no difference to",
        "does not bear on",
    )),
]

# --------------------------------------------------------------------------
# STRUCTURAL. Part-whole and connectivity. Transitive, mostly positive.
# --------------------------------------------------------------------------
STRUCTURAL: List[Relation] = [
    Relation("CONTAINS", "STRUCTURAL", POSITIVE, True, (
        "contains", "holds", "encloses", "houses", "includes",
        "comprises", "incorporates", "is made up of", "consists of",
        "encompasses", "carries", "accommodates", "bounds",
    )),
    Relation("PART_OF", "STRUCTURAL", POSITIVE, True, (
        "is part of", "belongs to", "is a component of", "is contained in",
        "is housed in", "sits within", "is an element of", "forms part of",
        "is a member of", "falls under", "is situated in", "resides in",
    )),
    Relation("CONNECTS_TO", "STRUCTURAL", POSITIVE, True, (
        "connects to", "is joined to", "links to", "attaches to",
        "is coupled to", "feeds into", "opens onto", "communicates with",
        "is tied to", "runs to", "is fastened to", "interfaces with",
        "is plumbed to", "is wired to",
    )),
    Relation("ISOLATED_FROM", "STRUCTURAL", NEGATIVE, True, (
        "is isolated from", "is disconnected from", "is separated from",
        "is sealed off from", "has no connection to", "is cut off from",
        "is detached from", "does not communicate with", "is decoupled from",
        "is partitioned from",
    )),
]

# --------------------------------------------------------------------------
# TEMPORAL. Ordering. Transitive.
# --------------------------------------------------------------------------
TEMPORAL: List[Relation] = [
    Relation("PRECEDES", "TEMPORAL", POSITIVE, True, (
        "precedes", "comes before", "occurs before", "happens prior to",
        "is followed by", "leads into", "predates", "antedates",
        "takes place ahead of", "runs before", "opens before",
    )),
    Relation("FOLLOWS", "TEMPORAL", NEGATIVE, True, (
        "follows", "comes after", "occurs after", "happens subsequent to",
        "is preceded by", "postdates", "trails", "succeeds",
        "takes place behind", "comes later than",
    )),
    Relation("CONCURRENT_WITH", "TEMPORAL", POSITIVE, False, (
        "occurs during", "coincides with", "happens at the same time as",
        "is simultaneous with", "overlaps", "runs alongside",
        "accompanies", "is concurrent with",
    )),
]

# --------------------------------------------------------------------------
# COMPARATIVE. Magnitude ordering. Transitive.
# --------------------------------------------------------------------------
COMPARATIVE: List[Relation] = [
    Relation("EXCEEDS", "COMPARATIVE", POSITIVE, True, (
        "exceeds", "is greater than", "is higher than", "surpasses",
        "outstrips", "is above", "tops", "overtakes", "is larger than",
        "runs higher than", "outweighs", "is in excess of",
    )),
    Relation("FALLS_BELOW", "COMPARATIVE", NEGATIVE, True, (
        "falls below", "is less than", "is lower than", "is under",
        "drops beneath", "is smaller than", "trails", "is short of",
        "runs lower than", "undershoots", "is beneath",
    )),
    Relation("MATCHES", "COMPARATIVE", POSITIVE, False, (
        "equals", "matches", "is the same as", "is equivalent to",
        "is on par with", "corresponds to", "is level with",
    )),
]

# --------------------------------------------------------------------------
# EVIDENTIAL. What a reading or record tells you. Not transitive.
# --------------------------------------------------------------------------
EVIDENTIAL: List[Relation] = [
    Relation("INDICATES", "EVIDENTIAL", POSITIVE, False, (
        "indicates", "shows", "reports", "registers", "signals",
        "reveals", "demonstrates", "confirms", "evidences", "attests to",
        "points to", "records", "reads", "displays", "suggests",
    )),
    Relation("CONTRADICTS", "EVIDENTIAL", NEGATIVE, False, (
        "contradicts", "disproves", "refutes", "is inconsistent with",
        "argues against", "tells against", "casts doubt on",
        "conflicts with", "belies", "rules against",
    )),
]

ALL_RELATIONS: List[Relation] = CAUSAL + STRUCTURAL + TEMPORAL + COMPARATIVE + EVIDENTIAL
RELATION_BY_NAME: Dict[str, Relation] = {r.name: r for r in ALL_RELATIONS}

# Phrase -> relation. Longest-match-first is enforced at lookup time so that
# "does not lead to" wins over "leads to".
PHRASE_TO_RELATION: Dict[str, Relation] = {}
for _r in ALL_RELATIONS:
    for _p in _r.phrases:
        PHRASE_TO_RELATION[_p] = _r

PHRASES_LONGEST_FIRST: List[str] = sorted(
    PHRASE_TO_RELATION, key=lambda p: (-len(p.split()), -len(p))
)


def compose(a: Relation, b: Relation) -> Tuple[str, int] | None:
    """Compose two relations along a chain.

    Returns (kind, polarity), or None when there is no product.

    None is the honest UNKNOWN: not low confidence, but an absence of any
    defined composition. Crossing kinds has no product, and neither does
    chaining through a non-transitive relation - knowing that a gauge
    indicates a pressure, and that the pressure exceeds a limit, does not tell
    you what the gauge stands in relation to the limit.
    """
    if a.kind != b.kind:
        return None
    if not (a.transitive and b.transitive):
        return None
    return (a.kind, a.polarity ^ b.polarity)


def compose_chain(relations: List[Relation]) -> Tuple[str, int] | None:
    if not relations:
        return None
    kind = relations[0].kind
    if any(r.kind != kind for r in relations):
        return None
    if any(not r.transitive for r in relations):
        return None
    polarity = 0
    for r in relations:
        polarity ^= r.polarity
    return (kind, polarity)


def stats() -> Dict[str, int]:
    by_kind: Dict[str, int] = {}
    for r in ALL_RELATIONS:
        by_kind[r.kind] = by_kind.get(r.kind, 0) + len(r.phrases)
    return {
        "relation_types": len(ALL_RELATIONS),
        "total_phrases": len(PHRASE_TO_RELATION),
        "kinds": len(KINDS),
        **{f"phrases_{k}": v for k, v in by_kind.items()},
    }
