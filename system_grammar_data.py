from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Dict, List, Tuple

ROLES = ["B", "P", "T", "G", "L", "R", "C", "D", "I"]
ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}

# Direction of one nominated terminal Level after the intervention.
DIR_NAMES = ["DOWN", "UNCHANGED", "UP", "INDETERMINATE"]
DOWN, UNCHANGED, UP, INDETERMINATE = range(4)

TG_NAMES = ["NEITHER", "TOPOLOGY_ONLY", "GEOMETRY_ONLY", "BOTH"]
NEITHER, TOPOLOGY_ONLY, GEOMETRY_ONLY, BOTH = range(4)

ACTION_NAMES = ["OBSERVATION", "INTERVENTION"]
OBSERVATION, INTERVENTION = range(2)

# Domain-independent causal grammar used by the synthetic world generator.
# The model is NOT given this graph directly.
DESCENDANTS = {
    "B": {"T", "R", "L"},
    "P": {"T", "G", "R", "L"},
    "T": {"R", "L"},
    "G": {"R", "L"},
    "L": set(),
    "R": {"L"},
    "C": {"R", "L"},
    "D": {"L"},
    "I": {"P", "T", "G", "R", "L"},
}

# Qualitative effect on the nominated terminal level for a positive change
# in each role. C and D use a convention explained in templates:
# +C = looser constraint; +D = shorter delay; +I = better information.
LEVEL_SIGN = {
    "B": +1,
    "P": +1,
    "T": +1,
    "G": +1,
    "L": +1,
    "R": +1,
    "C": +1,
    "D": +1,
    "I": +1,
}


@dataclass
class Intervention:
    roles: Dict[str, int]   # role -> +1 or -1
    text: str
    policy_subtype: str | None = None


@dataclass
class Case:
    domain: str
    facts: List[str]
    event_text: str
    action_kind: int
    delta: List[int]
    affected: List[int]
    invariant: List[int]
    topology_geometry: int
    direction: int
    compound_size: int
    meta: Dict[str, object]

    @property
    def full_text(self) -> str:
        return " ".join(self.facts + [self.event_text])


MECH_FILLERS = [
    "The maintenance log was printed on white paper.",
    "A technician arrived before sunrise.",
    "The equipment cabinet is painted gray.",
    "The control room contains two spare stools.",
    "A clipboard hangs beside the main panel.",
    "The site uses a weekly inspection schedule.",
]

ADMIN_FILLERS = [
    "The agency's letterhead was updated last year.",
    "A conference room was reserved for Friday.",
    "One memorandum used a blue cover page.",
    "The department has offices on two floors.",
    "The annual report contains several charts.",
    "A clerk copied the notice to an archive mailbox.",
]


def causal_closure(delta_roles: List[str]) -> List[str]:
    affected = set(delta_roles)
    frontier = list(delta_roles)
    while frontier:
        r = frontier.pop()
        for d in DESCENDANTS[r]:
            if d not in affected:
                affected.add(d)
                frontier.append(d)
    return [r for r in ROLES if r in affected]


def derive_direction(role_signs: Dict[str, int], action_kind: int) -> int:
    if action_kind == OBSERVATION:
        return INDETERMINATE
    score = sum(LEVEL_SIGN[r] * s for r, s in role_signs.items())
    if score > 0:
        return UP
    if score < 0:
        return DOWN
    return UNCHANGED


def tg_class(delta_roles: Dict[str, int]) -> int:
    t = "T" in delta_roles
    g = "G" in delta_roles
    if t and g:
        return BOTH
    if t:
        return TOPOLOGY_ONLY
    if g:
        return GEOMETRY_ONLY
    return NEITHER


def _vec(role_names: List[str]) -> List[int]:
    s = set(role_names)
    return [1 if r in s else 0 for r in ROLES]


def mechanical_base_facts(rng: random.Random) -> List[str]:
    a, b, c = rng.sample(["Alpha", "Beta", "Gamma", "Delta"], 3)
    return [
        f"Tank {a} is inside the controlled fluid network.",
        f"A pipe currently connects Tank {a} to Tank {b}.",
        f"The existing pipe has a rated capacity of {rng.choice([4,5,6,8])} liters per second.",
        f"Pump P currently moves water at {rng.choice([2,3,4])} liters per second.",
        f"Tank {b} presently contains {rng.choice([30,40,50,60])} liters.",
        f"A controller applies a rule to the valve between Tank {b} and Tank {c}.",
        f"A safety limit constrains the permitted flow.",
        f"The valve responds after a short actuation delay.",
        f"A level sensor reports Tank {b} to the controller.",
    ]


def admin_base_facts(rng: random.Random) -> List[str]:
    a, b, c = rng.sample(["Licensing Office", "Review Division", "Central Board", "Regional Unit"], 3)
    return [
        f"The {a} is inside the program's operative jurisdiction.",
        f"The {a} currently has authority to refer eligible matters to the {b}.",
        f"The existing referral relation has a capacity of {rng.choice([20,30,40,50])} matters per cycle.",
        f"The {a} currently processes {rng.choice([8,10,12,15])} matters per week.",
        f"The pending queue presently contains {rng.choice([40,60,80,100])} matters.",
        f"A governing rule determines when the {a} may activate the referral path to the {b}.",
        f"A statutory cap constrains the permitted processing volume.",
        f"A waiting period delays the effect of a completed referral.",
        f"A reporting system supplies queue information to the decision-maker.",
    ]


def mechanical_atomic(rng: random.Random, role: str, sign: int) -> Intervention:
    plus = sign > 0
    if role == "B":
        text = rng.choice([
            "Tank Delta is added to the controlled network." if plus else "Tank Delta is removed from the controlled network.",
            "The control boundary now includes Tank Delta." if plus else "The control boundary no longer includes Tank Delta.",
        ])
    elif role == "T":
        text = rng.choice([
            "A new pipe connection is opened from Tank Beta to Tank Gamma." if plus else "The pipe connection from Tank Beta to Tank Gamma is physically removed.",
            "Valve V establishes a new operative flow path between Beta and Gamma." if plus else "Valve V no longer provides any flow path between Beta and Gamma.",
        ])
    elif role == "G":
        text = rng.choice([
            "The capacity of the existing Beta-to-Gamma line is increased." if plus else "The capacity of the existing Beta-to-Gamma line is reduced.",
            "The existing valve's conductance setting is raised." if plus else "The existing valve's conductance setting is lowered.",
        ])
    elif role == "L":
        text = rng.choice([
            "Twenty liters are added directly to Tank Gamma." if plus else "Twenty liters are drained directly from Tank Gamma.",
            "The current liquid level in Tank Gamma is raised." if plus else "The current liquid level in Tank Gamma is lowered.",
        ])
    elif role == "R":
        text = rng.choice([
            "Pump P's actual transfer rate is increased." if plus else "Pump P's actual transfer rate is decreased.",
            "The current flow rate through the pump is raised." if plus else "The current flow rate through the pump is reduced.",
        ])
    elif role == "C":
        text = rng.choice([
            "The safety constraint is loosened so more flow is permitted." if plus else "The safety constraint is tightened so less flow is permitted.",
            "The operating cap is relaxed." if plus else "The operating cap is made more restrictive.",
        ])
    elif role == "D":
        text = rng.choice([
            "The valve's actuation delay is shortened." if plus else "The valve's actuation delay is lengthened.",
            "The response delay is reduced." if plus else "The response delay is increased.",
        ])
    elif role == "I":
        text = rng.choice([
            "The controller receives a more complete real-time sensor feed." if plus else "The controller loses access to the relevant sensor feed.",
            "Information available to the controller is improved." if plus else "The controller can no longer observe the relevant tank level.",
        ])
    elif role == "P":
        # Pure policy change that changes the rule, not topology or geometry by itself.
        text = rng.choice([
            "The controller policy is revised to favor earlier activation of the existing valve." if plus else "The controller policy is revised to delay activation of the existing valve.",
            "The governing control rule becomes more permissive." if plus else "The governing control rule becomes more restrictive.",
        ])
    else:
        raise ValueError(role)
    return Intervention({role: sign}, text)


def admin_atomic(rng: random.Random, role: str, sign: int) -> Intervention:
    plus = sign > 0
    if role == "B":
        text = rng.choice([
            "Class-X applications are added to the program's operative jurisdiction." if plus else "Class-X applications are removed from the program's operative jurisdiction.",
            "The program boundary now includes Class-X matters." if plus else "The program boundary no longer includes Class-X matters.",
        ])
    elif role == "T":
        text = rng.choice([
            "A new review authority is created from the Licensing Office to the Central Board." if plus else "The Licensing Office's authority to refer matters to the Central Board is abolished.",
            "An operative referral relation is added between the Licensing Office and Central Board." if plus else "The existing referral relation between the Licensing Office and Central Board is removed.",
        ])
    elif role == "G":
        text = rng.choice([
            "The capacity of the existing referral relation is increased." if plus else "The capacity of the existing referral relation is reduced.",
            "The maximum volume carried by the existing review channel is raised." if plus else "The maximum volume carried by the existing review channel is lowered.",
        ])
    elif role == "L":
        text = rng.choice([
            "Forty new matters are added directly to the pending queue." if plus else "Forty matters are removed directly from the pending queue.",
            "The current backlog level is increased." if plus else "The current backlog level is reduced.",
        ])
    elif role == "R":
        text = rng.choice([
            "The office's actual processing rate is increased." if plus else "The office's actual processing rate is decreased.",
            "The number of matters processed each week is raised." if plus else "The number of matters processed each week is reduced.",
        ])
    elif role == "C":
        text = rng.choice([
            "The statutory processing cap is loosened." if plus else "The statutory processing cap is tightened.",
            "The operative constraint is relaxed so more matters may be processed." if plus else "The operative constraint is made more restrictive.",
        ])
    elif role == "D":
        text = rng.choice([
            "The mandatory waiting period is shortened." if plus else "The mandatory waiting period is lengthened.",
            "The delay before a completed referral takes effect is reduced." if plus else "The delay before a completed referral takes effect is increased.",
        ])
    elif role == "I":
        text = rng.choice([
            "The decision-maker receives a more complete real-time queue report." if plus else "The decision-maker loses access to the relevant queue report.",
            "Information available to the decision-maker is improved." if plus else "The decision-maker can no longer observe the current backlog.",
        ])
    elif role == "P":
        text = rng.choice([
            "The governing policy is revised to favor earlier activation of the existing referral process." if plus else "The governing policy is revised to delay activation of the existing referral process.",
            "The governing rule becomes more permissive." if plus else "The governing rule becomes more restrictive.",
        ])
    else:
        raise ValueError(role)
    return Intervention({role: sign}, text)


def policy_structural_intervention(rng: random.Random, domain: str, subtype: str, sign: int) -> Intervention:
    """Policy intervention that also changes topology OR geometry, but not both."""
    plus = sign > 0
    if domain == "mechanical":
        if subtype == "topology":
            text = (
                "The controller policy is changed so Valve V now creates an operative Beta-to-Gamma flow path."
                if plus else
                "The controller policy is changed so Valve V is no longer permitted to create the Beta-to-Gamma flow path."
            )
            return Intervention({"P": sign, "T": sign}, text, "topology")
        else:
            text = (
                "The controller policy is changed so the existing valve may operate at a higher capacity without changing which tanks are connected."
                if plus else
                "The controller policy is changed so the existing valve must operate at a lower capacity without changing which tanks are connected."
            )
            return Intervention({"P": sign, "G": sign}, text, "geometry")
    else:
        if subtype == "topology":
            text = (
                "A new governing rule gives the Licensing Office authority to refer Class-X matters to the Central Board."
                if plus else
                "A new governing rule removes the Licensing Office's authority to refer Class-X matters to the Central Board."
            )
            return Intervention({"P": sign, "T": sign}, text, "topology")
        else:
            text = (
                "A new governing rule raises the capacity of the existing review channel without changing who may review whom."
                if plus else
                "A new governing rule lowers the capacity of the existing review channel without changing who may review whom."
            )
            return Intervention({"P": sign, "G": sign}, text, "geometry")


def make_case(
    rng: random.Random,
    domain: str,
    *,
    compound_size: int = 1,
    observation_probability: float = 0.15,
    force_policy_structural: bool = False,
) -> Case:
    if domain not in ("mechanical", "administrative"):
        raise ValueError(domain)

    facts = mechanical_base_facts(rng) if domain == "mechanical" else admin_base_facts(rng)
    fillers = MECH_FILLERS if domain == "mechanical" else ADMIN_FILLERS
    facts += rng.sample(fillers, k=rng.randint(1, 3))

    atom_fn = mechanical_atomic if domain == "mechanical" else admin_atomic

    interventions: List[Intervention] = []
    used_roles = set()

    if force_policy_structural:
        subtype = rng.choice(["topology", "geometry"])
        sign = rng.choice([-1, +1])
        interventions.append(policy_structural_intervention(rng, domain, subtype, sign))
        used_roles.update(interventions[-1].roles)
    else:
        for _ in range(compound_size):
            # Include structural policy interventions sometimes, even during normal generation.
            if rng.random() < 0.25 and "P" not in used_roles and compound_size == 1:
                subtype = rng.choice(["topology", "geometry"])
                sign = rng.choice([-1, +1])
                it = policy_structural_intervention(rng, domain, subtype, sign)
            else:
                available = [r for r in ROLES if r not in used_roles]
                role = rng.choice(available)
                sign = rng.choice([-1, +1])
                it = atom_fn(rng, role, sign)
            interventions.append(it)
            used_roles.update(it.roles)

    role_signs: Dict[str, int] = {}
    for it in interventions:
        for r, s in it.roles.items():
            role_signs[r] = max(-1, min(1, role_signs.get(r, 0) + s))

    role_signs = {r: s for r, s in role_signs.items() if s != 0}

    action_kind = OBSERVATION if rng.random() < observation_probability else INTERVENTION

    if action_kind == OBSERVATION:
        # Same surface event is reported as an observation rather than a deliberate intervention.
        event_text = "It is observed that " + " ".join(it.text[0].lower() + it.text[1:] for it in interventions)
        delta_roles = {}
    else:
        event_text = "Intervention: " + " ".join(it.text for it in interventions)
        delta_roles = role_signs

    affected_roles = causal_closure(list(delta_roles))
    invariant_roles = [r for r in ROLES if r not in set(affected_roles)]

    return Case(
        domain=domain,
        facts=facts,
        event_text=event_text,
        action_kind=action_kind,
        delta=_vec(list(delta_roles)),
        affected=_vec(affected_roles),
        invariant=_vec(invariant_roles),
        topology_geometry=tg_class(delta_roles),
        direction=derive_direction(delta_roles, action_kind),
        compound_size=compound_size,
        meta={
            "role_signs": delta_roles,
            "policy_subtypes": [it.policy_subtype for it in interventions if it.policy_subtype],
        },
    )


def generate_dataset(
    seed: int,
    n: int,
    domain: str,
    *,
    compound_sizes=(1,),
    observation_probability=0.15,
    policy_structural_fraction=0.20,
) -> List[Case]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        compound = rng.choice(compound_sizes)
        force_policy = compound == 1 and rng.random() < policy_structural_fraction
        out.append(
            make_case(
                rng,
                domain,
                compound_size=compound,
                observation_probability=observation_probability,
                force_policy_structural=force_policy,
            )
        )
    return out


TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,;:!?-]")


class Vocabulary:
    def __init__(self, cases: List[Case]):
        toks = set()
        for c in cases:
            for txt in c.facts + [c.event_text]:
                toks.update(self.tokenize(txt))
        self.itos = ["<PAD>", "<UNK>"] + sorted(toks)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @staticmethod
    def tokenize(text: str):
        return [x.lower() for x in TOKEN_RE.findall(text)]

    def encode_sentence(self, text: str, max_words: int):
        ids = [self.stoi.get(t, 1) for t in self.tokenize(text)][:max_words]
        return ids + [0] * (max_words - len(ids))

    def encode_flat(self, case: Case, max_tokens: int):
        ids = []
        for txt in case.facts + [case.event_text]:
            ids.extend(self.stoi.get(t, 1) for t in self.tokenize(txt))
        ids = ids[:max_tokens]
        return ids + [0] * (max_tokens - len(ids))
