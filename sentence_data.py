"""Experiment 6 — sentence-level relation induction.  Second design.

WHAT WENT WRONG THE FIRST TIME
    Version 1 trained on exactly one sentence shape -- "{a} {phrase} {b}" -- and
    then tested on four shapes the model had never encountered: passives,
    constructions with no relation word, and nominalizations. Four of the five
    test tiers asked the model to invent grammatical conventions it had never
    been shown once. Three of them scored below the majority-class floor, which
    is what an impossible task looks like.

    You cannot derive the English passive from first principles. Neither can a
    child; they hear it. The failure was in the split, not in the model.

WHAT CHANGED
    Every sentence shape now appears in training. What is held out is the
    *particular wording*, not the *kind of sentence*:

        shape         in training        held out for testing
        -----------   ----------------   ----------------------------------
        plain         most frames        some frames (tier B)
                      most phrases       the rest of the phrases (tier D)
        inflected     2/3 of relations   the other 1/3 (tier C)
        passive       2/3 of relations   the other 1/3 (tier F)
        construction  all but one each   the held-out one (tier E)
        nominal       all but one each   the held-out one (tier G)

    So the question becomes answerable: given a sentence pattern the model has
    seen used for other relations, can it read a relation it has not seen
    expressed that way? That is generalization. The old question -- can it read
    a sentence pattern it has never seen at all -- is not.

THE TIERS

    A  familiar        trained phrase, trained frame, new nonce entities.
                       Baseline. If this is not high, nothing else means
                       anything.

    B  new frame       trained phrase in a plain frame template held out of
                       training. Cheapest possible generalization.

    C  inflection      "will prevent", "is preventing" -- a trained phrase in
                       a form it was never shown, for a relation whose modal
                       and progressive frames were held out. The stem is
                       shared, so this is recoverable from characters.

    D  synonym         a held-out phrase that shares no stem with any trained
                       phrase for that relation: trained on "prevents", tested
                       on "blocks".

                       THIS IS A CONTROL, NOT A TARGET. The link between a
                       sound and a meaning is arbitrary -- Saussure's point --
                       so there is no signal in the characters of "blocks"
                       that says CAUSAL/NEGATIVE. This tier should fail. It is
                       here to mark the ceiling that arbitrariness imposes, so
                       that a failure elsewhere can be told apart from a
                       failure here.

    E  construction    a construction template held out, for a relation whose
                       other constructions were trained. No relation word from
                       the lexicon appears anywhere in the sentence.

                       THIS IS THE EXPERIMENT. Passing means the relation was
                       read out of sentence structure rather than looked up
                       from a word.

    F  passive         passive voice for a relation whose passives were held
                       out. Surface order of the two entities is reversed and
                       the relation direction is not. Tests whether the
                       passive convention, learned on other relations,
                       transfers.

    G  nominal         a noun-form template held out, for a relation whose
                       other noun forms were trained.

    Entities are nonce throughout, in training and in every tier. Real nouns
    would let the model key on topic association rather than on structure.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from relational_lexicon import (
    ALL_RELATIONS,
    KIND_INDEX,
    KINDS,
    PHRASE_TO_RELATION,
    Relation,
)

# --------------------------------------------------------------------------
# Nonce entities. Same construction as the morphology wug set.
# --------------------------------------------------------------------------
ONSETS = ["bl", "br", "dr", "fl", "gr", "kl", "pl", "sn", "st", "tr", "v", "z",
          "sp", "kr", "gl", "th", "sk", "pr", "fr", "sl", "shr", "tw", "qu"]
NUCLEI = ["a", "e", "i", "o", "u", "or", "ar", "ur", "ee", "oo", "ai", "ea"]
CODAS = ["b", "d", "g", "k", "m", "n", "p", "rb", "rk", "sk", "t", "v", "z",
         "nt", "ld", "mp", "ng", "st"]


def nonce(rng: random.Random) -> str:
    return rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)


def nonce_pair(rng: random.Random) -> Tuple[str, str]:
    """Two distinct nonce forms, neither a substring of the other."""
    for _ in range(60):
        a, b = nonce(rng), nonce(rng)
        if a != b and a not in b and b not in a:
            return a, b
    return "zorp", "blint"


def render(frame: str, rng: random.Random, **extra) -> Tuple[str, str, str] | None:
    """Fill a frame with nonce entities that each occur exactly once.

    Span marking finds an entity by string search, so the entity has to be
    findable and unique in the finished sentence. The syllable generator can
    and does emit real English words -- it once produced "than" for a frame
    containing "runs lower than" -- and it can emit a form that sits inside
    another word. Checking the rendered text catches every such case at once.
    """
    for _ in range(40):
        a, b = nonce_pair(rng)
        text = frame.format(a=a, b=b, **extra)
        low = text.lower()
        if low.count(a) == 1 and low.count(b) == 1:
            return text, a, b
    return None


# --------------------------------------------------------------------------
# English inflection. Enough to build "will prevent" and "is preventing" from
# the third-person forms the lexicon stores.
# --------------------------------------------------------------------------
VOWELS = "aeiou"

IRREGULAR_BARE: Dict[str, str] = {
    "is": "be", "has": "have", "does": "do", "goes": "go", "belies": "belie",
}

# "does not cause" is not a verb with a "does" head; the negation carries and
# the modal takes the whole thing. "would do not cause" is not English.
NEGATED = "does not "

# Two-syllable verbs stressed on the final syllable double their consonant
# before -ing. Stress is not derivable from spelling, so the ones this lexicon
# actually contains are listed.
DOUBLE_FINAL = {"admit", "occur", "outstrip", "overlap", "permit"}


def _syllables(word: str) -> int:
    n, prev_vowel = 0, False
    for ch in word:
        v = ch in VOWELS
        if v and not prev_vowel:
            n += 1
        prev_vowel = v
    return max(n, 1)


def bare_form(phrase: str) -> str | None:
    """Third-person singular -> bare infinitive. None if not inflectable."""
    if phrase.startswith(NEGATED):
        return "not " + phrase[len(NEGATED):]
    head, _, rest = phrase.partition(" ")
    if head in IRREGULAR_BARE:
        w = IRREGULAR_BARE[head]
    elif head.endswith("ies") and len(head) > 4:
        w = head[:-3] + "y"
    elif head.endswith(("sses", "shes", "ches", "xes", "zzes", "oes")):
        w = head[:-2]
    elif head.endswith("s") and not head.endswith("ss"):
        w = head[:-1]
    else:
        return None
    if not w:
        return None
    return (w + " " + rest).strip()


def _participle(bare: str) -> str:
    head, _, rest = bare.partition(" ")
    w = head
    if w == "be":
        w = "being"
    elif w.endswith("ie"):
        w = w[:-2] + "ying"
    elif w.endswith("e") and not w.endswith(("ee", "oe", "ye")):
        w = w[:-1] + "ing"
    elif (len(w) >= 3 and w[-1] not in VOWELS and w[-1] not in "wxy"
          and w[-2] in VOWELS and w[-3] not in VOWELS
          and (_syllables(w) == 1 or w in DOUBLE_FINAL)):
        w = w + w[-1] + "ing"
    else:
        w = w + "ing"
    return (w + " " + rest).strip()


def ing_form(phrase: str) -> str | None:
    """Bare infinitive -> present participle."""
    if phrase.startswith(NEGATED):
        return "not " + _participle(phrase[len(NEGATED):])
    bare = bare_form(phrase)
    if bare is None:
        return None
    return _participle(bare)


# --------------------------------------------------------------------------
# Frames, by shape.
# --------------------------------------------------------------------------
PLAIN_FRAMES = [
    "{a} {p} {b}.",
    "The {a} {p} the {b}.",
    "In this system the {a} {p} the {b}.",
    "It is established that {a} {p} {b}.",
    "Under normal operation the {a} {p} the {b}.",
    "Records show that {a} {p} {b}.",
    "Each {a} {p} some {b}.",
    "Whenever it is present, the {a} {p} the {b}.",
    "Across every trial the {a} {p} the {b}.",
    "By all accounts {a} {p} {b}.",
    "Here the {a} {p} the {b}.",
    "As a rule, {a} {p} {b}.",
]

MODAL_FRAMES = [
    "{a} will {v} {b}.",
    "The {a} may {v} the {b}.",
    "It is known that {a} can {v} {b}.",
    "Any {a} would {v} some {b}.",
    "Under load the {a} should {v} the {b}.",
]

PROGRESSIVE_FRAMES = [
    "The {a} is {v} the {b}.",
    "{a} is currently {v} {b}.",
    "The {a} keeps {v} the {b}.",
]

# Passive. Arguments swap on the surface; the relation does not.
PARTICIPLES: Dict[str, str] = {
    "CAUSES": "caused", "PREVENTS": "prevented", "ENABLES": "enabled",
    "REQUIRES": "required", "INCREASES": "increased", "DECREASES": "decreased",
    "CONTAINS": "contained", "CONNECTS_TO": "connected to",
    "PRECEDES": "preceded", "EXCEEDS": "exceeded", "INDICATES": "indicated",
    "CONTRADICTS": "contradicted", "ISOLATED_FROM": "isolated from",
    "PART_OF": "included in", "FOLLOWS": "followed",
}

PASSIVE_FRAMES = [
    "{b} is {pp} by {a}.",
    "The {b} is {pp} by the {a}.",
    "It is reported that {b} is {pp} by {a}.",
    "Every {b} is {pp} by some {a}.",
    "In each case the {b} is {pp} by the {a}.",
]

# --------------------------------------------------------------------------
# Constructions. No lexicon phrase appears anywhere in these.
# --------------------------------------------------------------------------
CONSTRUCTIONS: Dict[str, List[str]] = {
    "CAUSES": [
        "{b} is downstream of {a}.",
        "{a} is why {b} happens.",
        "{b} is a consequence of {a}.",
        "Where {a} is present, {b} appears.",
        "{a}, and so {b}.",
        "{b} on account of {a}.",
        "Given {a}, {b} ensues.",
        "{a} is behind {b}.",
    ],
    "PREVENTS": [
        "{b} does not occur where {a} is present.",
        "{a} keeps {b} from happening.",
        "With {a} in place, {b} never appears.",
        "{a} stands in the way of {b}.",
        "So long as {a} holds, {b} is off the table.",
        "{b} has no chance while {a} is there.",
    ],
    "REQUIRES": [
        "Without {b} there is no {a}.",
        "{a} is impossible in the absence of {b}.",
        "No {a} occurs unless {b} is present.",
        "{b} is a precondition of {a}.",
        "Take away {b} and {a} is gone.",
    ],
    "INCREASES": [
        "More {a} means more {b}.",
        "As {a} rises, so does {b}.",
        "{b} climbs with {a}.",
        "The greater the {a}, the greater the {b}.",
        "Where {a} runs high, {b} runs high too.",
    ],
    "DECREASES": [
        "More {a} means less {b}.",
        "As {a} rises, {b} falls.",
        "{b} drops off as {a} grows.",
        "The greater the {a}, the smaller the {b}.",
        "Where {a} runs high, {b} runs low.",
    ],
    "CONTAINS": [
        "Inside every {a} there is a {b}.",
        "{b} lies in the middle of {a}.",
        "You will find {b} inside {a}.",
        "{a} has {b} in it.",
        "Look within {a} and there is {b}.",
    ],
    "PRECEDES": [
        "{a} first, then {b}.",
        "{b} only after {a}.",
        "By the time {b} happens, {a} already has.",
        "{a} comes first and {b} later.",
        "First {a}; {b} afterward.",
    ],
    "EXCEEDS": [
        "{a} is bigger than {b}.",
        "{b} does not reach {a}.",
        "{a} sits over {b}.",
        "Nothing about {b} reaches the level of {a}.",
        "Set {a} beside {b} and the winner is clear.",
    ],
    "ENABLES": [
        "With {a} in hand, {b} becomes possible.",
        "{a} opens the door to {b}.",
        "{b} is within reach once {a} is there.",
        "{a} clears the path for {b}.",
        "Once {a} is in place, {b} can go ahead.",
    ],
    "FAILS_TO_PRODUCE": [
        "{a} does nothing at all to {b}.",
        "Whether {a} is there or not, {b} is the same.",
        "{b} is untouched by {a}.",
        "{a} leaves {b} exactly as it was.",
        "Nothing about {b} turns on {a}.",
    ],
    "PART_OF": [
        "{a} is one piece of {b}.",
        "You would count {a} among the pieces of {b}.",
        "{a} makes up some of {b}.",
        "{b} would not be whole without {a}.",
    ],
    "CONNECTS_TO": [
        "There is a line from {a} to {b}.",
        "{a} and {b} are hooked together.",
        "A path runs from {a} through to {b}.",
        "You can get from {a} to {b} directly.",
    ],
    "ISOLATED_FROM": [
        "Nothing runs between {a} and {b}.",
        "There is no path at all from {a} to {b}.",
        "{a} and {b} never touch.",
        "You cannot get from {a} to {b} at all.",
    ],
    "FOLLOWS": [
        "{b} first, then {a}.",
        "{a} only after {b}.",
        "{a} turns up once {b} is done.",
        "First {b}; {a} afterward.",
    ],
    "CONCURRENT_WITH": [
        "{a} and {b} happen together.",
        "While {a} is going on, so is {b}.",
        "{a} and {b} run side by side.",
        "One moment covers both {a} and {b}.",
    ],
    "FALLS_BELOW": [
        "{a} never gets as big as {b}.",
        "{a} does not come up to {b}.",
        "{a} sits under {b}.",
        "Set {a} beside {b} and {a} comes up short.",
    ],
    "MATCHES": [
        "{a} and {b} come to the same thing.",
        "There is nothing between {a} and {b}.",
        "{a} is neither more nor less than {b}.",
        "Measure {a}, measure {b}, no difference.",
    ],
    "INDICATES": [
        "From {a} you can read off {b}.",
        "{a} tells you about {b}.",
        "{a} is a sign of {b}.",
        "Watch {a} and you learn {b}.",
    ],
    "CONTRADICTS": [
        "{a} cannot be squared with {b}.",
        "{a} and {b} cannot both hold.",
        "{a} counts as evidence the other way from {b}.",
        "If {a} is right then {b} is wrong.",
    ],
}

# --------------------------------------------------------------------------
# Nominalizations. Relation carried by a noun. Three each so that holding one
# out still leaves two in training.
# --------------------------------------------------------------------------
NOMINALIZATIONS: Dict[str, List[str]] = {
    "CAUSES": [
        "The cause of {b} is {a}.",
        "{a} is the reason for {b}.",
        "The origin of {b} lies in {a}.",
    ],
    "PREVENTS": [
        "The barrier to {b} is {a}.",
        "{a} is the obstacle to {b}.",
        "Protection from {b} comes from {a}.",
    ],
    "REQUIRES": [
        "The prerequisite for {a} is {b}.",
        "{b} is a requirement of {a}.",
        "The condition on {a} is {b}.",
    ],
    "CONTAINS": [
        "The contents of {a} include {b}.",
        "{b} is among the parts of {a}.",
        "The interior of {a} is where {b} sits.",
    ],
    "INCREASES": [
        "The driver of {b} is {a}.",
        "{a} is the source of growth in {b}.",
        "The lift on {b} comes from {a}.",
    ],
    "DECREASES": [
        "The drag on {b} is {a}.",
        "{a} is the source of decline in {b}.",
        "The damper on {b} is {a}.",
    ],
    "PRECEDES": [
        "The forerunner of {b} is {a}.",
        "{a} is the earlier of the two.",
        "The one before {b} is {a}.",
    ],
    "EXCEEDS": [
        "The larger of the two is {a}, not {b}.",
        "{a} is the upper of the two, over {b}.",
        "The higher reading is {a}, not {b}.",
    ],
    "INDICATES": [
        "The signal for {b} is {a}.",
        "{a} is the readout of {b}.",
        "The evidence for {b} is {a}.",
    ],
    "CONNECTS_TO": [
        "The link to {b} is {a}.",
        "{a} is the route to {b}.",
        "The junction with {b} is {a}.",
    ],
    "ENABLES": [
        "The enabler of {b} is {a}.",
        "{a} is the opening for {b}.",
        "The permission for {b} comes from {a}.",
    ],
    "PART_OF": [
        "The whole containing {a} is {b}.",
        "{a} is a constituent of {b}.",
        "The larger body over {a} is {b}.",
    ],
    "FOLLOWS": [
        "The successor of {b} is {a}.",
        "{a} is the later of the two.",
        "The one after {b} is {a}.",
    ],
    "CONTRADICTS": [
        "The counterevidence to {b} is {a}.",
        "{a} is the objection to {b}.",
        "The case against {b} is built from {a}.",
    ],
    "FALLS_BELOW": [
        "The smaller of the two is {a}, not {b}.",
        "{a} is the lower of the two, under {b}.",
        "The lower reading is {a}, not {b}.",
    ],
}

# --------------------------------------------------------------------------
# Tiers.
# --------------------------------------------------------------------------
TIERS: List[Tuple[str, str]] = [
    ("A_familiar", "Trained wording and frame, new entities"),
    ("B_new_frame", "Held-out plain frame, trained relation word"),
    ("C_inflection", "Inflected trained word, relation held out of inflected frames"),
    ("D_synonym", "Held-out synonym (arbitrariness control -- expected to fail)"),
    ("E_construction", "Held-out construction, no relation word present"),
    ("F_passive", "Passive for a relation whose passives were held out"),
    ("G_nominal", "Held-out noun-form phrasing"),
]
TIER_NAMES = [t for t, _ in TIERS]
TIER_DESCRIPTIONS = dict(TIERS)

SHAPES = ["plain", "modal", "progressive", "passive", "construction", "nominal"]


@dataclass
class Sample:
    """One sentence with two marked entity spans and a relation target."""
    text: str
    source: str            # the entity the relation runs FROM
    target: str            # the entity the relation runs TO
    relation: str          # relation name
    kind: int
    polarity: int
    tier: str              # "train" or a tier code
    shape: str
    frame: str
    phrase_present: bool   # a lexicon phrase appears verbatim
    stem_shared: bool      # a trained phrase's stem appears in some form

    def describe(self) -> str:
        return (f"[{self.tier:14s}/{self.shape:12s}] {self.text}\n"
                f"      {self.source} --{self.relation}--> {self.target}   "
                f"kind={KINDS[self.kind]} pol={self.polarity} "
                f"phrase={self.phrase_present} stem={self.stem_shared}")


def _make(text, src, tgt, rel: Relation, tier, shape, frame,
          phrase_present, stem_shared) -> Sample:
    return Sample(text=text, source=src, target=tgt, relation=rel.name,
                  kind=KIND_INDEX[rel.kind], polarity=rel.polarity,
                  tier=tier, shape=shape, frame=frame,
                  phrase_present=phrase_present, stem_shared=stem_shared)


def _split(items: Sequence, rng: random.Random, n_held: int) -> Tuple[List, List]:
    xs = list(items)
    rng.shuffle(xs)
    n_held = max(0, min(n_held, len(xs) - 1))
    return xs[n_held:], xs[:n_held]


def _split_relations(names: Sequence[str], rng: random.Random,
                     n_held: int) -> Tuple[List[str], List[str]]:
    """Hold out relations spread across (kind, polarity) classes.

    A plain random draw can hand the test tier five relations that share two
    classes between them, which pushes the majority-class floor up near the
    accuracy a real result would sit at. Round-robin over classes keeps the
    floor low enough that the tier can be read.
    """
    by_name = {r.name: r for r in ALL_RELATIONS}
    buckets: Dict[Tuple[str, int], List[str]] = {}
    for n in names:
        r = by_name[n]
        buckets.setdefault((r.kind, r.polarity), []).append(n)
    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets, key=lambda k: (-len(buckets[k]), str(k)))
    held: List[str] = []
    i = 0
    while len(held) < n_held and any(buckets.values()):
        b = buckets[order[i % len(order)]]
        if b:
            held.append(b.pop())
        i += 1
    train = [n for n in names if n not in held]
    return train, held


class SentenceGenerator:
    """Holds every split. Fixed by seed, so train and test never overlap on
    the thing being held out, and every *shape* is present on both sides."""

    def __init__(self, seed: int = 71, phrase_holdout: float = 0.25):
        rng = random.Random(seed)

        # -- plain frames -------------------------------------------------
        self.train_frames, self.held_frames = _split(PLAIN_FRAMES, rng, 3)

        # -- phrases, per relation. Held-out phrases are distinct lemmas, so
        #    this is the arbitrariness control, not a solvable tier.
        self.train_phrases: Dict[str, List[str]] = {}
        self.held_phrases: Dict[str, List[str]] = {}
        for r in ALL_RELATIONS:
            n_held = max(1, int(len(r.phrases) * phrase_holdout))
            tr, hl = _split(r.phrases, rng, n_held)
            self.train_phrases[r.name] = tr
            self.held_phrases[r.name] = hl

        # -- which relations see modal/progressive frames in training -----
        names = [r.name for r in ALL_RELATIONS]
        self.inflect_train_rels, self.inflect_test_rels = _split_relations(
            names, rng, max(1, len(names) // 3))

        # -- which relations see passives in training ---------------------
        pass_names = [n for n in names if n in PARTICIPLES]
        self.passive_train_rels, self.passive_test_rels = _split_relations(
            pass_names, rng, max(1, len(pass_names) // 3))

        # -- constructions and nominals: hold one template out per relation
        self.train_constructions: Dict[str, List[str]] = {}
        self.held_constructions: Dict[str, List[str]] = {}
        for name, opts in CONSTRUCTIONS.items():
            tr, hl = _split(opts, rng, 1 if len(opts) >= 3 else 0)
            self.train_constructions[name] = tr
            self.held_constructions[name] = hl

        self.train_nominals: Dict[str, List[str]] = {}
        self.held_nominals: Dict[str, List[str]] = {}
        for name, opts in NOMINALIZATIONS.items():
            tr, hl = _split(opts, rng, 1 if len(opts) >= 3 else 0)
            self.train_nominals[name] = tr
            self.held_nominals[name] = hl

        self.by_name = {r.name: r for r in ALL_RELATIONS}

    # -- builders ---------------------------------------------------------
    def _plain(self, rng, rel, phrases, frames, tier) -> Sample | None:
        if not phrases or not frames:
            return None
        p = rng.choice(phrases)
        frame = rng.choice(frames)
        r = render(frame, rng, p=p)
        if r is None:
            return None
        text, a, b = r
        return _make(text, a, b, rel, tier, "plain", frame, True, True)

    def _inflected(self, rng, rel, tier) -> Sample | None:
        phrases = self.train_phrases[rel.name]
        rng.shuffle(phrases := list(phrases))
        for p in phrases:
            use_prog = rng.random() < 0.4 and not p.startswith("is ")
            v = ing_form(p) if use_prog else bare_form(p)
            if v is None:
                continue
            frame = rng.choice(PROGRESSIVE_FRAMES if use_prog else MODAL_FRAMES)
            r = render(frame, rng, v=v)
            if r is None:
                continue
            text, a, b = r
            shape = "progressive" if use_prog else "modal"
            return _make(text, a, b, rel, tier, shape, frame, False, True)
        return None

    def _passive(self, rng, rel, tier) -> Sample | None:
        pp = PARTICIPLES.get(rel.name)
        if pp is None:
            return None
        frame = rng.choice(PASSIVE_FRAMES)
        r = render(frame, rng, pp=pp)
        if r is None:
            return None
        # Surface order is b-then-a; the relation still runs a -> b.
        text, a, b = r
        return _make(text, a, b, rel, tier, "passive", frame, False, True)

    def _from_templates(self, rng, rel, templates, tier, shape) -> Sample | None:
        opts = templates.get(rel.name)
        if not opts:
            return None
        frame = rng.choice(opts)
        r = render(frame, rng)
        if r is None:
            return None
        text, a, b = r
        return _make(text, a, b, rel, tier, shape, frame, False, False)

    # -- training set -----------------------------------------------------
    def training_set(self, seed: int, count: int) -> List[Sample]:
        """Every shape, mixed. Weighted toward plain because that is where the
        relation vocabulary is actually taught; the other shapes are there so
        the conventions are learnable at all."""
        rng = random.Random(seed)
        weights = [("plain", 0.40), ("inflected", 0.12), ("passive", 0.14),
                   ("construction", 0.22), ("nominal", 0.12)]
        shapes = [s for s, _ in weights]
        probs = [w for _, w in weights]
        out: List[Sample] = []
        guard = 0
        while len(out) < count and guard < count * 60:
            guard += 1
            shape = rng.choices(shapes, probs)[0]
            rel = self.by_name[rng.choice(list(self.by_name))]
            s = None
            if shape == "plain":
                s = self._plain(rng, rel, self.train_phrases[rel.name],
                                self.train_frames, "train")
            elif shape == "inflected":
                if rel.name in self.inflect_train_rels:
                    s = self._inflected(rng, rel, "train")
            elif shape == "passive":
                if rel.name in self.passive_train_rels:
                    s = self._passive(rng, rel, "train")
            elif shape == "construction":
                s = self._from_templates(rng, rel, self.train_constructions,
                                         "train", "construction")
            elif shape == "nominal":
                s = self._from_templates(rng, rel, self.train_nominals,
                                         "train", "nominal")
            if s is not None:
                out.append(s)
        return out

    # -- test tiers -------------------------------------------------------
    def tier(self, seed: int, count: int, tier: str) -> List[Sample]:
        rng = random.Random(seed)
        out: List[Sample] = []
        guard = 0
        pool = list(self.by_name.values())
        while len(out) < count and guard < count * 120:
            guard += 1
            rel = rng.choice(pool)
            s = None
            if tier == "A_familiar":
                s = self._plain(rng, rel, self.train_phrases[rel.name],
                                self.train_frames, tier)
            elif tier == "B_new_frame":
                s = self._plain(rng, rel, self.train_phrases[rel.name],
                                self.held_frames, tier)
            elif tier == "C_inflection":
                if rel.name in self.inflect_test_rels:
                    s = self._inflected(rng, rel, tier)
            elif tier == "D_synonym":
                s = self._plain(rng, rel, self.held_phrases[rel.name],
                                self.train_frames, tier)
                if s is not None:
                    s.stem_shared = False
            elif tier == "E_construction":
                s = self._from_templates(rng, rel, self.held_constructions,
                                         tier, "construction")
            elif tier == "F_passive":
                if rel.name in self.passive_test_rels:
                    s = self._passive(rng, rel, tier)
                    if s is not None:
                        s.stem_shared = False
            elif tier == "G_nominal":
                s = self._from_templates(rng, rel, self.held_nominals,
                                         tier, "nominal")
            else:
                raise ValueError(f"unknown tier {tier}")
            if s is not None:
                out.append(s)
        return out

    # -- composition ------------------------------------------------------
    def chains(self, seed: int, count: int, tier: str) -> List[Chain]:
        """Pairs of same-kind transitive sentences joined at a middle entity.

        Both halves are drawn from the same tier, so composition can be tested
        on unseen constructions as well as on familiar wording.
        """
        rng = random.Random(seed)
        out: List[Chain] = []
        guard = 0
        while len(out) < count and guard < count * 200:
            guard += 1
            left = self.tier(rng.randrange(1 << 30), 1, tier)
            if not left:
                continue
            first = left[0]
            r1 = self.by_name[first.relation]
            if not r1.transitive:
                continue
            right = self.tier(rng.randrange(1 << 30), 1, tier)
            if not right:
                continue
            second = right[0]
            r2 = self.by_name[second.relation]
            if not r2.transitive or r2.kind != r1.kind:
                continue
            # Rename the second sentence's source to the first's target so the
            # two sentences meet at one entity.
            mid = first.target
            if mid == second.target or mid in second.target or second.target in mid:
                continue
            text = second.text.replace(second.source, mid)
            if text.lower().count(mid.lower()) != 1:
                continue
            second = Sample(**{**second.__dict__, "text": text, "source": mid})
            out.append(Chain(first=first, second=second,
                             kind=KIND_INDEX[r1.kind],
                             polarity=r1.polarity ^ r2.polarity,
                             defined=True))
        return out

    # -- audit ------------------------------------------------------------
    def split_report(self) -> Dict[str, object]:
        return {
            "plain_frames": {"train": len(self.train_frames),
                             "held_out": len(self.held_frames)},
            "inflected_relations": {"train": len(self.inflect_train_rels),
                                    "held_out": len(self.inflect_test_rels),
                                    "held_out_names": sorted(self.inflect_test_rels)},
            "passive_relations": {"train": len(self.passive_train_rels),
                                  "held_out": len(self.passive_test_rels),
                                  "held_out_names": sorted(self.passive_test_rels)},
            "phrases": {r.name: {"train": len(self.train_phrases[r.name]),
                                 "held_out": len(self.held_phrases[r.name])}
                        for r in ALL_RELATIONS},
            "constructions": {n: {"train": len(self.train_constructions[n]),
                                  "held_out": len(self.held_constructions[n])}
                              for n in CONSTRUCTIONS},
            "nominals": {n: {"train": len(self.train_nominals[n]),
                             "held_out": len(self.held_nominals[n])}
                         for n in NOMINALIZATIONS},
        }


@dataclass
class Chain:
    """Two sentences sharing a middle entity.

        first:   a --R1--> b
        second:  b --R2--> c

    The composed relation a --> c is R1 . R2, which within a kind is
    polarity = p1 XOR p2. This is the sentence-level version of the
    composition that worked at word level: the model reads each sentence
    separately and the two answers are added, never learned jointly.
    """
    first: Sample
    second: Sample
    kind: int
    polarity: int
    defined: bool


# --------------------------------------------------------------------------
# Leak audit. Tiers C, E, F and G must contain no lexicon phrase verbatim;
# otherwise the model can look the relation up instead of reading it.
# --------------------------------------------------------------------------
PHRASE_LIST = sorted(PHRASE_TO_RELATION, key=len, reverse=True)

# Word boundaries matter here. "is undershooting" contains the characters of
# "is under" but a reader is not looking a phrase up when it sees it -- the
# word is a different word. Substring matching would report leaks that are not
# leaks and hide the ones that are.
_PHRASE_RE = {p: re.compile(r"(?<!\w)" + re.escape(p) + r"(?!\w)")
              for p in PHRASE_LIST}


def find_leaks(samples: Sequence[Sample]) -> List[Tuple[str, str]]:
    hits = []
    for s in samples:
        low = s.text.lower()
        for p in PHRASE_LIST:
            if _PHRASE_RE[p].search(low):
                hits.append((s.text, p))
                break
    return hits


def audit(gen: "SentenceGenerator", seed: int = 999, n: int = 400) -> Dict[str, object]:
    """Every check that has to hold before the experiment means anything."""
    report: Dict[str, object] = {}
    train = gen.training_set(seed, n * 2)
    report["train_shape_counts"] = _counts(s.shape for s in train)
    for name in TIER_NAMES:
        ds = gen.tier(seed + hash(name) % 1000, n, name)
        entry: Dict[str, object] = {
            "n": len(ds),
            "shapes": _counts(s.shape for s in ds),
            "relations": len({s.relation for s in ds}),
            "class_floor": _floor(train, ds),
        }
        if name in ("C_inflection", "E_construction", "F_passive", "G_nominal"):
            leaks = find_leaks(ds)
            entry["lexicon_phrase_leaks"] = len(leaks)
            entry["leak_examples"] = leaks[:5]
        bad = [s for s in ds
               if s.text.lower().count(s.source.lower()) != 1
               or s.text.lower().count(s.target.lower()) != 1]
        entry["span_findable"] = len(ds) - len(bad)
        if bad:
            entry["ambiguous_span_frames"] = sorted({s.frame for s in bad})
        report[name] = entry
    return report


def _counts(it) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in it:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items()))


def _floor(train: Sequence[Sample], test: Sequence[Sample]) -> float:
    from collections import Counter
    if not train or not test:
        return 0.0
    top = Counter((s.kind, s.polarity) for s in train).most_common(1)[0][0]
    return sum(1 for s in test if (s.kind, s.polarity) == top) / len(test)


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
def shuffle_words(sample: Sample, rng: random.Random) -> Sample:
    """Destroy syntax, keep vocabulary. If accuracy survives this, the model is
    doing bag-of-words and the structural claim is dead."""
    words = sample.text.split()
    rng.shuffle(words)
    return Sample(**{**sample.__dict__, "text": " ".join(words)})


def swap_spans(sample: Sample) -> Sample:
    """Swap which span is source and which is target. A model that scores the
    same on this as on the original is not reading direction at all."""
    d = dict(sample.__dict__)
    d["source"], d["target"] = sample.target, sample.source
    return Sample(**d)


class CharVocabulary:
    """Characters only. No word list anywhere."""

    def __init__(self, samples: Sequence[Sample]):
        chars = set()
        for s in samples:
            chars.update(s.text.lower())
            chars.update(s.source.lower())
            chars.update(s.target.lower())
        self.itos = ["<pad>"] + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, text: str, max_chars: int) -> List[int]:
        ids = [self.stoi.get(ch, 0) for ch in text.lower()[:max_chars]]
        return ids + [0] * (max_chars - len(ids))


if __name__ == "__main__":
    import json
    g = SentenceGenerator()
    print(json.dumps(audit(g), indent=2, default=str))
