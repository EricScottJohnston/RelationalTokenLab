"""Experiment 1b — Language-internal relational structure.

Ground truth here never refers to the world. A chain like

    nation -> national -> nationalize -> nationalization

is scored purely on form-to-form relations. You do not need to know what a
nation is; you need to know that the pair (nation, national) stands in the same
relation as (form, formal) and (person, personal).

Two composable relation systems are tracked, both language-internal:

  CATEGORY   a transition over {NOUN, VERB, ADJ, ADV}. Composes by function
             composition: N->ADJ followed by ADJ->V is N->V.

  POLARITY   a Z2 group. Negation is an involution: un-un-X == X. Composes by
             XOR, exactly like the Z4 phases in Experiment 1 but with order 2.

The surface realization of a relation varies heavily inside each language
(PLURAL is -s / -es / -en / vowel change; PAST is -ed or ablaut) and completely
across languages. That variation is the point: the relation is supposed to be
recoverable from structural position, not from the letters.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

CATEGORIES = ["NOUN", "VERB", "ADJ", "ADV"]
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

# Derivational operations. Each carries the category transition it performs and
# whether it flips polarity. Nothing here encodes meaning.
OPERATIONS = [
    "ADJECTIVAL",     # NOUN -> ADJ      nation / national
    "CAUSATIVE",      # ADJ  -> VERB     national / nationalize
    "NOMINALIZE",     # VERB -> NOUN     nationalize / nationalization
    "AGENTIVE",       # VERB -> NOUN     teach / teacher
    "ADVERBIAL",      # ADJ  -> ADV      quick / quickly
    "NEGATION",       # category-preserving involution   happy / unhappy
    "PLURAL",         # NOUN -> NOUN     teacher / teachers
    "PAST",           # VERB -> VERB     teach / taught
]
OP_INDEX = {o: i for i, o in enumerate(OPERATIONS)}

OP_TRANSITION: Dict[str, Tuple[str, str]] = {
    "ADJECTIVAL": ("NOUN", "ADJ"),
    "CAUSATIVE": ("ADJ", "VERB"),
    "NOMINALIZE": ("VERB", "NOUN"),
    "AGENTIVE": ("VERB", "NOUN"),
    "ADVERBIAL": ("ADJ", "ADV"),
    "NEGATION": ("ANY", "SAME"),
    "PLURAL": ("NOUN", "NOUN"),
    "PAST": ("VERB", "VERB"),
}
POLARITY_FLIPPING = {"NEGATION"}


@dataclass(frozen=True)
class Entry:
    """One derivational step: source form, operation, resulting form."""
    source: str
    operation: str
    result: str
    source_cat: str
    result_cat: str


# ---------------------------------------------------------------------------
# Lexicons. Hand-built so ground truth is exact.
#
# Irregulars and suppletion are deliberately included (go/went, good/better,
# mouse/mice). Those are the cases where the surface gives you nothing at all
# and only structural position identifies the relation.
# ---------------------------------------------------------------------------

ENGLISH: List[Tuple[str, str, str]] = [
    # NOUN -> ADJ
    ("nation", "ADJECTIVAL", "national"),
    ("form", "ADJECTIVAL", "formal"),
    ("person", "ADJECTIVAL", "personal"),
    ("industry", "ADJECTIVAL", "industrial"),
    ("critic", "ADJECTIVAL", "critical"),
    ("history", "ADJECTIVAL", "historical"),
    ("centre", "ADJECTIVAL", "central"),
    ("culture", "ADJECTIVAL", "cultural"),
    ("music", "ADJECTIVAL", "musical"),
    ("nature", "ADJECTIVAL", "natural"),
    # ADJ -> VERB
    ("national", "CAUSATIVE", "nationalize"),
    ("formal", "CAUSATIVE", "formalize"),
    ("personal", "CAUSATIVE", "personalize"),
    ("industrial", "CAUSATIVE", "industrialize"),
    ("central", "CAUSATIVE", "centralize"),
    ("legal", "CAUSATIVE", "legalize"),
    ("modern", "CAUSATIVE", "modernize"),
    ("stable", "CAUSATIVE", "stabilize"),
    ("private", "CAUSATIVE", "privatize"),
    ("special", "CAUSATIVE", "specialize"),
    ("general", "CAUSATIVE", "generalize"),
    ("real", "CAUSATIVE", "realize"),
    # VERB -> NOUN (nominalization)
    ("nationalize", "NOMINALIZE", "nationalization"),
    ("formalize", "NOMINALIZE", "formalization"),
    ("personalize", "NOMINALIZE", "personalization"),
    ("industrialize", "NOMINALIZE", "industrialization"),
    ("centralize", "NOMINALIZE", "centralization"),
    ("legalize", "NOMINALIZE", "legalization"),
    ("modernize", "NOMINALIZE", "modernization"),
    ("stabilize", "NOMINALIZE", "stabilization"),
    ("privatize", "NOMINALIZE", "privatization"),
    ("specialize", "NOMINALIZE", "specialization"),
    ("generalize", "NOMINALIZE", "generalization"),
    ("develop", "NOMINALIZE", "development"),
    ("govern", "NOMINALIZE", "government"),
    ("manage", "NOMINALIZE", "management"),
    ("employ", "NOMINALIZE", "employment"),
    ("create", "NOMINALIZE", "creation"),
    ("produce", "NOMINALIZE", "production"),
    ("act", "NOMINALIZE", "action"),
    # VERB -> NOUN (agentive)
    ("teach", "AGENTIVE", "teacher"),
    ("write", "AGENTIVE", "writer"),
    ("govern", "AGENTIVE", "governor"),
    ("employ", "AGENTIVE", "employer"),
    ("manage", "AGENTIVE", "manager"),
    ("create", "AGENTIVE", "creator"),
    ("produce", "AGENTIVE", "producer"),
    ("act", "AGENTIVE", "actor"),
    ("paint", "AGENTIVE", "painter"),
    ("sing", "AGENTIVE", "singer"),
    # ADJ -> ADV
    ("quick", "ADVERBIAL", "quickly"),
    ("national", "ADVERBIAL", "nationally"),
    ("formal", "ADVERBIAL", "formally"),
    ("critical", "ADVERBIAL", "critically"),
    ("natural", "ADVERBIAL", "naturally"),
    ("central", "ADVERBIAL", "centrally"),
    ("legal", "ADVERBIAL", "legally"),
    ("special", "ADVERBIAL", "specially"),
    # NEGATION (allomorphy: un- / im- / il- / ir- / in-)
    ("happy", "NEGATION", "unhappy"),
    ("possible", "NEGATION", "impossible"),
    ("legal", "NEGATION", "illegal"),
    ("regular", "NEGATION", "irregular"),
    ("responsible", "NEGATION", "irresponsible"),
    ("logical", "NEGATION", "illogical"),
    ("relevant", "NEGATION", "irrelevant"),
    ("mature", "NEGATION", "immature"),
    ("formal", "NEGATION", "informal"),
    ("personal", "NEGATION", "impersonal"),
    ("stable", "NEGATION", "unstable"),
    ("natural", "NEGATION", "unnatural"),
    # PLURAL (allomorphy + suppletion)
    ("teacher", "PLURAL", "teachers"),
    ("writer", "PLURAL", "writers"),
    ("nation", "PLURAL", "nations"),
    ("church", "PLURAL", "churches"),
    ("box", "PLURAL", "boxes"),
    ("child", "PLURAL", "children"),
    ("mouse", "PLURAL", "mice"),
    ("foot", "PLURAL", "feet"),
    ("goose", "PLURAL", "geese"),
    ("man", "PLURAL", "men"),
    ("woman", "PLURAL", "women"),
    ("ox", "PLURAL", "oxen"),
    # PAST (regular + ablaut + suppletion)
    ("teach", "PAST", "taught"),
    ("write", "PAST", "wrote"),
    ("paint", "PAST", "painted"),
    ("sing", "PAST", "sang"),
    ("go", "PAST", "went"),
    ("act", "PAST", "acted"),
    ("create", "PAST", "created"),
    ("bring", "PAST", "brought"),
    ("think", "PAST", "thought"),
    ("employ", "PAST", "employed"),
    # Continuations so chains reach depth 4-5. Nominalizations pluralize;
    # negated adjectives take adverbial marking.
    ("nationalization", "PLURAL", "nationalizations"),
    ("formalization", "PLURAL", "formalizations"),
    ("personalization", "PLURAL", "personalizations"),
    ("centralization", "PLURAL", "centralizations"),
    ("legalization", "PLURAL", "legalizations"),
    ("modernization", "PLURAL", "modernizations"),
    ("stabilization", "PLURAL", "stabilizations"),
    ("privatization", "PLURAL", "privatizations"),
    ("specialization", "PLURAL", "specializations"),
    ("generalization", "PLURAL", "generalizations"),
    ("development", "PLURAL", "developments"),
    ("government", "PLURAL", "governments"),
    ("creation", "PLURAL", "creations"),
    ("production", "PLURAL", "productions"),
    ("action", "PLURAL", "actions"),
    ("governor", "PLURAL", "governors"),
    ("employer", "PLURAL", "employers"),
    ("manager", "PLURAL", "managers"),
    ("creator", "PLURAL", "creators"),
    ("producer", "PLURAL", "producers"),
    ("actor", "PLURAL", "actors"),
    ("painter", "PLURAL", "painters"),
    ("singer", "PLURAL", "singers"),
    ("illegal", "ADVERBIAL", "illegally"),
    ("informal", "ADVERBIAL", "informally"),
    ("impersonal", "ADVERBIAL", "impersonally"),
    ("unnatural", "ADVERBIAL", "unnaturally"),
    ("unhappy", "ADVERBIAL", "unhappily"),
    ("irregular", "ADVERBIAL", "irregularly"),
    ("illogical", "ADVERBIAL", "illogically"),
    ("unstable", "NEGATION", "stable"),
    ("informal", "CAUSATIVE", "informalize"),
    ("informalize", "NOMINALIZE", "informalization"),
    ("informalization", "PLURAL", "informalizations"),
]

GERMAN: List[Tuple[str, str, str]] = [
    # NOUN -> ADJ
    ("Nation", "ADJECTIVAL", "national"),
    ("Form", "ADJECTIVAL", "formal"),
    ("Person", "ADJECTIVAL", "persönlich"),
    ("Industrie", "ADJECTIVAL", "industriell"),
    ("Kritik", "ADJECTIVAL", "kritisch"),
    ("Geschichte", "ADJECTIVAL", "geschichtlich"),
    ("Zentrum", "ADJECTIVAL", "zentral"),
    ("Kultur", "ADJECTIVAL", "kulturell"),
    ("Musik", "ADJECTIVAL", "musikalisch"),
    ("Natur", "ADJECTIVAL", "natürlich"),
    # ADJ -> VERB
    ("national", "CAUSATIVE", "nationalisieren"),
    ("formal", "CAUSATIVE", "formalisieren"),
    ("industriell", "CAUSATIVE", "industrialisieren"),
    ("zentral", "CAUSATIVE", "zentralisieren"),
    ("legal", "CAUSATIVE", "legalisieren"),
    ("modern", "CAUSATIVE", "modernisieren"),
    ("stabil", "CAUSATIVE", "stabilisieren"),
    ("privat", "CAUSATIVE", "privatisieren"),
    ("speziell", "CAUSATIVE", "spezialisieren"),
    ("global", "CAUSATIVE", "globalisieren"),
    # VERB -> NOUN
    ("nationalisieren", "NOMINALIZE", "Nationalisierung"),
    ("formalisieren", "NOMINALIZE", "Formalisierung"),
    ("industrialisieren", "NOMINALIZE", "Industrialisierung"),
    ("zentralisieren", "NOMINALIZE", "Zentralisierung"),
    ("legalisieren", "NOMINALIZE", "Legalisierung"),
    ("modernisieren", "NOMINALIZE", "Modernisierung"),
    ("stabilisieren", "NOMINALIZE", "Stabilisierung"),
    ("privatisieren", "NOMINALIZE", "Privatisierung"),
    ("spezialisieren", "NOMINALIZE", "Spezialisierung"),
    ("globalisieren", "NOMINALIZE", "Globalisierung"),
    ("entwickeln", "NOMINALIZE", "Entwicklung"),
    ("regieren", "NOMINALIZE", "Regierung"),
    ("verwalten", "NOMINALIZE", "Verwaltung"),
    # VERB -> NOUN (agentive, -er / -in)
    ("lehren", "AGENTIVE", "Lehrer"),
    ("schreiben", "AGENTIVE", "Schreiber"),
    ("regieren", "AGENTIVE", "Regierer"),
    ("arbeiten", "AGENTIVE", "Arbeiter"),
    ("malen", "AGENTIVE", "Maler"),
    ("singen", "AGENTIVE", "Sänger"),
    ("denken", "AGENTIVE", "Denker"),
    ("spielen", "AGENTIVE", "Spieler"),
    # ADJ -> ADV: German uses the bare adjective. The relation is real; the
    # surface marking is ZERO. This is the sharpest cross-linguistic test in
    # the set — English suffixes -ly, German does nothing at all.
    ("schnell", "ADVERBIAL", "schnell"),
    ("national", "ADVERBIAL", "national"),
    ("formal", "ADVERBIAL", "formal"),
    ("kritisch", "ADVERBIAL", "kritisch"),
    ("natürlich", "ADVERBIAL", "natürlich"),
    ("zentral", "ADVERBIAL", "zentral"),
    ("legal", "ADVERBIAL", "legal"),
    # NEGATION (un- / in- / ir-)
    ("glücklich", "NEGATION", "unglücklich"),
    ("möglich", "NEGATION", "unmöglich"),
    ("legal", "NEGATION", "illegal"),
    ("regelmäßig", "NEGATION", "unregelmäßig"),
    ("logisch", "NEGATION", "unlogisch"),
    ("relevant", "NEGATION", "irrelevant"),
    ("reif", "NEGATION", "unreif"),
    ("formal", "NEGATION", "informal"),
    ("stabil", "NEGATION", "instabil"),
    ("natürlich", "NEGATION", "unnatürlich"),
    ("persönlich", "NEGATION", "unpersönlich"),
    ("klar", "NEGATION", "unklar"),
    # PLURAL (umlaut, -e, -er, -en, -s, zero)
    ("Lehrer", "PLURAL", "Lehrer"),
    ("Arbeiter", "PLURAL", "Arbeiter"),
    ("Nation", "PLURAL", "Nationen"),
    ("Kirche", "PLURAL", "Kirchen"),
    ("Kind", "PLURAL", "Kinder"),
    ("Maus", "PLURAL", "Mäuse"),
    ("Fuß", "PLURAL", "Füße"),
    ("Mann", "PLURAL", "Männer"),
    ("Frau", "PLURAL", "Frauen"),
    ("Buch", "PLURAL", "Bücher"),
    ("Auto", "PLURAL", "Autos"),
    ("Haus", "PLURAL", "Häuser"),
    # PAST (weak -te, strong ablaut, suppletion)
    ("lehren", "PAST", "lehrte"),
    ("schreiben", "PAST", "schrieb"),
    ("malen", "PAST", "malte"),
    ("singen", "PAST", "sang"),
    ("gehen", "PAST", "ging"),
    ("arbeiten", "PAST", "arbeitete"),
    ("denken", "PAST", "dachte"),
    ("bringen", "PAST", "brachte"),
    ("spielen", "PAST", "spielte"),
    ("sein", "PAST", "war"),
    # Continuations for depth 4-5.
    ("Nationalisierung", "PLURAL", "Nationalisierungen"),
    ("Formalisierung", "PLURAL", "Formalisierungen"),
    ("Industrialisierung", "PLURAL", "Industrialisierungen"),
    ("Zentralisierung", "PLURAL", "Zentralisierungen"),
    ("Legalisierung", "PLURAL", "Legalisierungen"),
    ("Modernisierung", "PLURAL", "Modernisierungen"),
    ("Stabilisierung", "PLURAL", "Stabilisierungen"),
    ("Privatisierung", "PLURAL", "Privatisierungen"),
    ("Spezialisierung", "PLURAL", "Spezialisierungen"),
    ("Globalisierung", "PLURAL", "Globalisierungen"),
    ("Entwicklung", "PLURAL", "Entwicklungen"),
    ("Regierung", "PLURAL", "Regierungen"),
    ("Verwaltung", "PLURAL", "Verwaltungen"),
    ("Schreiber", "PLURAL", "Schreiber"),
    ("Maler", "PLURAL", "Maler"),
    ("Sänger", "PLURAL", "Sänger"),
    ("Denker", "PLURAL", "Denker"),
    ("Spieler", "PLURAL", "Spieler"),
    ("illegal", "ADVERBIAL", "illegal"),
    ("informal", "ADVERBIAL", "informal"),
    ("unnatürlich", "ADVERBIAL", "unnatürlich"),
    ("unglücklich", "ADVERBIAL", "unglücklich"),
    ("unlogisch", "ADVERBIAL", "unlogisch"),
    ("instabil", "NEGATION", "stabil"),
    ("informal", "CAUSATIVE", "informalisieren"),
    ("informalisieren", "NOMINALIZE", "Informalisierung"),
    ("Informalisierung", "PLURAL", "Informalisierungen"),
]

LEXICONS = {"english": ENGLISH, "german": GERMAN}


def _category_of_step(op: str, incoming_cat: str | None) -> Tuple[str, str]:
    src, dst = OP_TRANSITION[op]
    if src == "ANY":
        cat = incoming_cat or "ADJ"
        return cat, cat
    return src, dst


def build_entries(language: str) -> List[Entry]:
    out: List[Entry] = []
    for source, op, result in LEXICONS[language]:
        src_cat, dst_cat = _category_of_step(op, None)
        out.append(Entry(source, op, result, src_cat, dst_cat))
    return out


@dataclass
class Chain:
    """A derivational path plus the composed relation between its endpoints."""
    language: str
    forms: List[str]                 # length depth+1
    operations: List[str]            # length depth
    start_cat: str
    end_cat: str
    polarity: int                    # 0 even negations, 1 odd
    cycle_advance: int               # net steps around the N->ADJ->VERB cycle
    depth: int
    pairs: List[Tuple[str, str]] = field(default_factory=list)

    def composed_relation(self) -> int:
        """Endpoint-to-endpoint relation: Z3 cycle advance x Z2 polarity.

        The productive derivational operations all step the same direction
        around a three-cycle:

            NOUN --ADJECTIVAL--> ADJ --CAUSATIVE--> VERB --NOMINALIZE--> NOUN

        so nation -> national -> nationalize -> nationalization returns to
        NOUN having advanced by three. Category-preserving operations (PLURAL,
        PAST) are the identity. Negation is an involution on a separate Z2.

        The result is a closed abelian group of order six. Unlike a raw
        start-to-end category pair, every class is reachable at every depth,
        which is what makes depth extrapolation measurable at all - the same
        property that made Z4 work in Experiment 1.
        """
        return (self.cycle_advance % 3) * 2 + self.polarity


NUM_COMPOSED_RELATIONS = 6

# Position in the productive derivational cycle. ADV is not on the cycle.
CYCLE = {"NOUN": 0, "ADJ": 1, "VERB": 2}
CYCLE_ADVANCING = {"ADJECTIVAL", "CAUSATIVE", "NOMINALIZE", "AGENTIVE"}
CYCLE_IDENTITY = {"PLURAL", "PAST", "NEGATION"}


class ChainGenerator:
    """Builds derivational chains by walking the lexicon graph."""

    def __init__(self, language: str):
        self.language = language
        self.entries = build_entries(language)
        self.by_source: Dict[str, List[Entry]] = {}
        for e in self.entries:
            self.by_source.setdefault(e.source, []).append(e)

    def sample(self, rng: random.Random, depth: int, max_tries: int = 200) -> Chain | None:
        for _ in range(max_tries):
            start = rng.choice(self.entries).source
            forms = [start]
            ops: List[str] = []
            cats: List[str] = []
            cur = start
            cur_cat: str | None = None
            ok = True
            for _ in range(depth):
                options = self.by_source.get(cur)
                if not options:
                    ok = False
                    break
                e = rng.choice(options)
                # Category continuity: a step must start where the last ended.
                if cur_cat is not None and e.source_cat != "ANY" and e.source_cat != cur_cat:
                    ok = False
                    break
                # Reject degenerate runs: three identical operations in a row
                # (e.g. German zero-plural repeated) carry no information.
                if len(ops) >= 2 and ops[-1] == ops[-2] == e.operation:
                    ok = False
                    break
                # ADV is off the productive cycle and is a dead end, so chains
                # that enter it cannot continue composing. Keep them out.
                if e.operation == "ADVERBIAL":
                    ok = False
                    break
                src_cat, dst_cat = _category_of_step(e.operation, cur_cat)
                if cur_cat is None:
                    cur_cat = src_cat
                    cats.append(src_cat)
                ops.append(e.operation)
                forms.append(e.result)
                cur_cat = dst_cat
                cats.append(dst_cat)
                cur = e.result
            if not ok or len(ops) != depth:
                continue
            polarity = sum(1 for o in ops if o in POLARITY_FLIPPING) % 2
            advance = sum(1 for o in ops if o in CYCLE_ADVANCING)
            chain = Chain(
                language=self.language,
                forms=forms,
                operations=ops,
                start_cat=cats[0],
                end_cat=cats[-1],
                polarity=polarity,
                cycle_advance=advance,
                depth=depth,
                pairs=[(forms[i], forms[i + 1]) for i in range(depth)],
            )
            return chain
        return None

    def dataset(self, seed: int, count: int, depths: Tuple[int, ...]) -> List[Chain]:
        rng = random.Random(seed)
        out: List[Chain] = []
        guard = 0
        while len(out) < count and guard < count * 60:
            guard += 1
            c = self.sample(rng, rng.choice(depths))
            if c is not None:
                out.append(c)
        return out


# ---------------------------------------------------------------------------
# Nonce stems — the wug test.
#
# Berko 1958: show a child a picture of a "wug", then two of them, and they say
# "wugs" without ever having heard the word. The rule cannot have been
# memorized, because the word does not exist.
#
# Real-word chains leave a loophole: "nationalization" could be one memorized
# string mapping to one answer. A nonce stem closes it. Every test item is a
# word that has never existed, so the only way to score is to decompose the
# form and compose the relations.
#
# Limitation worth stating: nonce stems can only take regular morphology. You
# cannot invent a suppletive. The irregulars (go/went, mouse/mice) live in the
# real-word test set, and the nonce set is regular by necessity.
# ---------------------------------------------------------------------------

ONSETS = ["bl", "br", "dr", "fl", "gr", "kl", "pl", "sn", "st", "tr", "v", "z",
          "sp", "kr", "gl", "th", "sk", "pr", "fr", "sl"]
NUCLEI = ["a", "e", "i", "o", "u", "or", "ar", "ur", "ee", "oo"]
CODAS = ["b", "d", "g", "k", "m", "n", "p", "rb", "rk", "sk", "t", "v", "z", "nt", "ld"]

REGULAR_AFFIXES = {
    "english": {
        "ADJECTIVAL": ("", "al"),
        "CAUSATIVE": ("", "ize"),
        "NOMINALIZE": ("", "ation"),
        "AGENTIVE": ("", "er"),
        "NEGATION": ("un", ""),
        "PLURAL": ("", "s"),
        "PAST": ("", "ed"),
    },
    "german": {
        "ADJECTIVAL": ("", "isch"),
        "CAUSATIVE": ("", "isieren"),
        "NOMINALIZE": ("", "ung"),
        "AGENTIVE": ("", "er"),
        "NEGATION": ("un", ""),
        "PLURAL": ("", "en"),
        "PAST": ("", "te"),
    },
}


def _nonce_stem(rng: random.Random) -> str:
    return rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)


def _apply_affix(form: str, op: str, language: str) -> str:
    pre, suf = REGULAR_AFFIXES[language][op]
    base = form
    # Light orthographic tidying so forms stay pronounceable rather than
    # introducing a spurious cue the model could key on.
    if suf and base.endswith("e") and suf[0] in "aeiou":
        base = base[:-1]
    if suf == "ation" and base.endswith("ize"):
        base = base[:-1]
    return pre + base + suf


class NonceGenerator:
    """Invented stems carrying real, regular morphology."""

    def __init__(self, language: str):
        self.language = language

    def sample(self, rng: random.Random, depth: int) -> Chain | None:
        stem = _nonce_stem(rng)
        cat = "NOUN"
        forms = [stem]
        ops: List[str] = []
        cur = stem
        for _ in range(depth):
            legal = []
            for op in OPERATIONS:
                if op == "ADVERBIAL":
                    continue
                src, _dst = OPERATION_SOURCE.get(op, (None, None))
                if src is None or src == "ANY" or src == cat:
                    legal.append(op)
            if not legal:
                return None
            if len(ops) >= 2 and ops[-1] == ops[-2]:
                legal = [o for o in legal if o != ops[-1]] or legal
            op = rng.choice(legal)
            cur = _apply_affix(cur, op, self.language)
            src, dst = OPERATION_SOURCE[op]
            cat = cat if dst == "SAME" else dst
            ops.append(op)
            forms.append(cur)
        return Chain(
            language=f"{self.language}-nonce",
            forms=forms,
            operations=ops,
            start_cat="NOUN",
            end_cat=cat,
            polarity=sum(1 for o in ops if o in POLARITY_FLIPPING) % 2,
            cycle_advance=sum(1 for o in ops if o in CYCLE_ADVANCING),
            depth=depth,
            pairs=[(forms[i], forms[i + 1]) for i in range(depth)],
        )

    def dataset(self, seed: int, count: int, depths: Tuple[int, ...]) -> List[Chain]:
        rng = random.Random(seed)
        out: List[Chain] = []
        guard = 0
        while len(out) < count and guard < count * 40:
            guard += 1
            c = self.sample(rng, rng.choice(depths))
            if c is not None:
                out.append(c)
        return out


OPERATION_SOURCE = dict(OP_TRANSITION)


class CharVocabulary:
    """Character-level only. No word list, no embeddings, no semantics.

    A word form reaches the model as its characters. Nothing about what it
    denotes is available anywhere in the pipeline.
    """

    def __init__(self, chains: List[Chain]):
        chars = set()
        for c in chains:
            for f in c.forms:
                chars.update(f.lower())
        self.itos = ["<pad>"] + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, form: str, max_chars: int = 24) -> List[int]:
        ids = [self.stoi.get(ch, 0) for ch in form.lower()[:max_chars]]
        return ids + [0] * (max_chars - len(ids))


def describe(chain: Chain) -> str:
    arrow = "  ".join(
        f"{chain.forms[i]} --{chain.operations[i]}--> {chain.forms[i+1]}"
        for i in range(chain.depth)
    )
    return (
        f"[{chain.language}] {arrow}\n"
        f"    composed: {chain.start_cat} -> {chain.end_cat}, "
        f"polarity={chain.polarity}, class={chain.composed_relation()}"
    )
