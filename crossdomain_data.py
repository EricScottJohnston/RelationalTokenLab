from __future__ import annotations
from dataclasses import dataclass
import random, re
from typing import List, Tuple, Dict

ADVERSE, NON_ADVERSE, UNRESOLVED = 0, 1, 2
TODDLER_LABELS = ['NOT_OK','OK','UNCLEAR']
LEGAL_LABELS = ['CONVERSION_LIKE','NOT_CONVERSION_LIKE','INDETERMINATE']

@dataclass
class Story:
    facts: List[str]
    label: int
    meta: Dict[str, object]
    @property
    def text(self): return ' '.join(self.facts)

A_NAMES=['Mia','Emma','Ava','Lily','Sophie','Nora']
B_NAMES=['Leo','Noah','Eli','Owen','Max','Finn']
TOYS=['wooden truck','stuffed rabbit','red ball','toy dinosaur','blue wagon','box of crayons']
T_IRR=['The playroom rug has stars on it.','Someone left a cup near the window.','It is sunny outside.','A teacher is wearing a green sweater.','Music is playing quietly.','There are four chairs by the wall.','Lunch today is pasta.','A backpack is beside the door.']
P_NAMES=['Orion Manufacturing','Northstar Fabrication','Pioneer Instruments','Atlas Tooling']
D_NAMES=['Mercer Holdings','Riverton Equipment','Summit Resale','Canyon Industrial']
AGENTS=['Delta Storage','Kepler Logistics','Apex Brokerage','Mesa Custody']
SUBAGENTS=['Harbor Transit','Vega Warehousing','Juniper Services','Redstone Handling']
ITEMS=['precision press','inspection scanner','machining spindle','calibration fixture','robotic toolhead']
L_IRR=['The invoice used a blue header.','The companies have offices in different counties.','One manager drove a silver sedan.','The warehouse exterior is tan.','The purchase order used twelve-point type.','A meeting occurred on a rainy morning.','The equipment crate had a white shipping label.','One email copied an accounting assistant.','The warehouse also stores unrelated furniture.','The parties used different payroll vendors.']

def toddler_story(rng, owner=None, a_permission=None, b_authorized=None, late_reveal=False, distractors=2):
    a,b,toy=rng.choice(A_NAMES),rng.choice(B_NAMES),rng.choice(TOYS)
    if a_permission is None: a_permission=rng.choice(['active','expired','none'])
    if b_authorized is None: b_authorized=rng.random()<0.18
    prov=[]
    if owner=='A': prov.append(rng.choice([f'{a} received the {toy} as a birthday present.',f'The {toy} belongs to {a}.',f'{a} brought her own {toy} from home.']))
    elif owner=='B': prov.append(rng.choice([f'{b} received the {toy} as a birthday present.',f'The {toy} belongs to {b}.',f'{b} brought his own {toy} from home.']))
    else: prov.append(rng.choice([f'No one knows whether the {toy} belongs to {a} or {b}.',f'The adults cannot tell who owns the {toy}.',f'It is unclear whether {a} or {b} has the better claim to the {toy}.']))
    if owner=='B':
        if a_permission=='active': prov.append(rng.choice([f'{b} told {a} she could use the {toy} until the bell rang, and the bell has not rung.',f'{b} promised {a} she could keep playing with the {toy} for ten more minutes, and only two minutes have passed.',f'{b} lent the {toy} to {a} for the rest of playtime, and playtime is still underway.']))
        elif a_permission=='expired': prov.append(rng.choice([f'{b} had lent the {toy} to {a} until the bell rang, and the bell already rang.',f"{a}'s agreed turn with the {toy} has ended.",f'The time {b} gave {a} to use the {toy} has expired.']))
        else: prov.append(rng.choice([f'{b} never said {a} could keep the {toy}.',f'{a} had taken the {toy} from {b} without asking.',f'{a} was holding the {toy} without permission from {b}.']))
    if owner=='A':
        if b_authorized: prov.append(rng.choice([f'{a} told {b} he could take the {toy} now.',f'{a} gave {b} permission to carry the {toy} away.',f'{a} said it was {b}\'s turn to take the {toy}.']))
        else: prov.append(rng.choice([f'{a} did not give {b} permission to take the {toy}.',f'{b} had not been told he could take the {toy}.',f'{a} asked to keep the {toy} with her.']))
    event=rng.choice([f'{b} takes the {toy} from {a} and keeps it.',f'{b} grabs the {toy} from {a} and walks away with it.',f"{b} removes the {toy} from {a}'s hands and keeps control of it."])
    irr=rng.sample(T_IRR,k=min(distractors,len(T_IRR)))
    facts=[event]+irr+prov if late_reveal else prov+irr+[event]
    if owner=='A': label=NON_ADVERSE if b_authorized else ADVERSE
    elif owner=='B': label=ADVERSE if a_permission=='active' else NON_ADVERSE
    else: label=UNRESOLVED
    return Story(facts,label,{'domain':'toddler','owner':owner,'a_permission':a_permission,'b_authorized':b_authorized,'late_reveal':late_reveal})

def legal_story(rng, chain_depth=2, plaintiff_right=None, authority=None, serious=None, distractors=5, late_reveal=False, forced_reveal=None):
    p,d,a1,a2,item=rng.choice(P_NAMES),rng.choice(D_NAMES),rng.choice(AGENTS),rng.choice(SUBAGENTS),rng.choice(ITEMS)
    chain_depth=max(1,min(4,chain_depth))
    if plaintiff_right is None: plaintiff_right=rng.choices(['yes','no','unknown'],weights=[.55,.30,.15])[0]
    if authority is None: authority=rng.choices(['valid','invalid','unknown'],weights=[.35,.50,.15])[0]
    if serious is None: serious=rng.random()<.82
    facts=[rng.choice([f'{p} originally acquired the {item}.',f'{p} purchased and took title to the {item}.',f'The {item} was initially owned by {p}.'])]
    prov=[]
    if plaintiff_right=='yes': prov.append(rng.choice([f'No valid sale of the {item} by {p} occurred before the disputed control.',f'{p} retained its right to immediate possession of the {item}.',f'At the time of the disputed act, {p} had not transferred its right to possess the {item}.']))
    elif plaintiff_right=='no': prov.append(rng.choice([f'Before the disputed act, {p} completed a valid sale transferring the present right to the {item}.',f'{p} had already transferred its immediate possessory right in the {item} before the dispute.',f'A completed authorized transaction ended {p}\'s right to immediate possession before the disputed act.']))
    else: prov.append(rng.choice([f'The record does not establish whether {p} retained the immediate right to possess the {item}.',f'It is unresolved whether an earlier transaction transferred {p}\'s present possessory right.',f'The facts do not determine who held the immediate right to possess the {item} at the disputed time.']))
    prov.append(rng.choice([f'{p} delivered the {item} to {a1} for limited custody.',f'{a1} received the {item} from {p} under a restricted custody arrangement.',f'{p} placed the {item} with {a1} for a limited commercial purpose.']))
    if chain_depth>=2: prov.append(rng.choice([f'{a1} then placed the {item} with {a2} for handling.',f'{a1} delegated physical custody of the {item} to {a2}.',f'{a2} later received the {item} through {a1}.']))
    if chain_depth>=3: prov.append(rng.choice([f'A written instruction limited downstream handling to storage and transport.',f'The custody chain contained an additional restriction against disposition.',f'Downstream possession remained subject to the original scope limits.']))
    if chain_depth>=4: prov.append(rng.choice([f'A later operational notice changed which intermediary could release the {item}.',f'The parties issued a supplemental instruction governing downstream release of the {item}.',f'A later custody instruction controlled whether the {item} could be released onward.']))
    if forced_reveal=='valid_authority': authority='valid'; auth=f'Before the disputed act, {p} expressly authorized the downstream transfer that placed the {item} under {d}\'s control.'
    elif forced_reveal=='revoked_authority': authority='invalid'; auth=f'Before the disputed act, {p} had revoked the authority on which the downstream transfer of the {item} to {d} depended.'
    elif authority=='valid': auth=rng.choice([f'The authority chain remained valid and within scope when the {item} reached {d}.',f'{p}\'s operative authorization permitted the transfer that gave {d} control of the {item}.',f'Every required authorization for {d}\'s control of the {item} was still effective.'])
    elif authority=='invalid': auth=rng.choice([f'Before {d} obtained control, a required authority in the chain had been revoked.',f'The transfer to {d} exceeded the still-operative scope of the custody authority.',f'A necessary authorization had expired before the {item} was placed under {d}\'s control.'])
    else: auth=rng.choice([f'The evidence does not establish whether the authority required for {d}\'s control was still operative.',f'It is unresolved whether the downstream transfer to {d} fell within valid authority.',f'The record does not determine whether the authority chain remained effective when {d} obtained the {item}.'])
    prov.append(auth)
    if forced_reveal=='plaintiff_sold': plaintiff_right='no'; prov.append(f'A newly produced agreement shows that {p} had validly transferred its immediate possessory right before {d}\'s disputed act.')
    control=rng.choice([f'{d} intentionally refused a demand to surrender the {item} and continued to control it as its own.',f'{d} intentionally sold the {item} onward despite {p}\'s demand for its return.',f'{d} intentionally retained exclusive control of the {item} after being told to return it.']) if serious else rng.choice([f'{d} briefly moved the {item} for safekeeping and immediately made it available for return.',f'{d} handled the {item} momentarily but did not retain, sell, or exclude anyone from control.',f'{d} temporarily repositioned the {item} and promptly relinquished control.'])
    irr=rng.sample(L_IRR,k=min(distractors,len(L_IRR)))
    facts=facts+[control]+irr+[x for x in prov if x!=auth]+[auth] if late_reveal else facts+prov+irr+[control]
    if not serious or plaintiff_right=='no' or (plaintiff_right=='yes' and authority=='valid'): label=NON_ADVERSE
    elif plaintiff_right=='yes' and authority=='invalid': label=ADVERSE
    else: label=UNRESOLVED
    return Story(facts,label,{'domain':'legal','plaintiff_right':plaintiff_right,'authority':authority,'serious':serious,'chain_depth':chain_depth,'late_reveal':late_reveal})

def generate_toddler_dataset(seed,n,late_fraction=.25):
    rng=random.Random(seed); out=[]
    for _ in range(n):
        owner=rng.choices(['A','B',None],weights=[.43,.43,.14])[0]
        out.append(toddler_story(rng,owner=owner,late_reveal=rng.random()<late_fraction,distractors=rng.randint(1,4)))
    return out

def generate_legal_dataset(seed,n,max_chain_depth,min_chain_depth=1,late_fraction=.25,distractor_range=(3,8)):
    rng=random.Random(seed); out=[]
    for _ in range(n): out.append(legal_story(rng,chain_depth=rng.randint(min_chain_depth,max_chain_depth),late_reveal=rng.random()<late_fraction,distractors=rng.randint(*distractor_range)))
    return out

def legal_material_pairs(seed,n_pairs,chain_depth=4):
    rng=random.Random(seed); out=[]
    for _ in range(n_pairs):
        s=rng.randrange(10**9)
        out.append((legal_story(random.Random(s),chain_depth=chain_depth,plaintiff_right='yes',authority='valid',serious=True,distractors=6),legal_story(random.Random(s),chain_depth=chain_depth,plaintiff_right='yes',authority='invalid',serious=True,distractors=6)))
    return out

def legal_irrelevant_pairs(seed,n_pairs,chain_depth=4):
    rng=random.Random(seed); out=[]
    for _ in range(n_pairs):
        right=rng.choice(['yes','no']); auth=rng.choice(['valid','invalid']); serious=rng.random()<.8
        out.append((legal_story(random.Random(rng.randrange(10**9)),chain_depth=chain_depth,plaintiff_right=right,authority=auth,serious=serious,distractors=8),legal_story(random.Random(rng.randrange(10**9)),chain_depth=chain_depth,plaintiff_right=right,authority=auth,serious=serious,distractors=8)))
    return out

def legal_revelation_pairs(seed,n_pairs,chain_depth=4):
    rng=random.Random(seed); out=[]
    for _ in range(n_pairs):
        resolution=rng.choice(['valid_authority','revoked_authority']); s=rng.randrange(10**9)
        initial=legal_story(random.Random(s),chain_depth=chain_depth,plaintiff_right='yes',authority='unknown',serious=True,distractors=6)
        initial.facts=[f for f in initial.facts if not any(k in f.lower() for k in ['does not establish whether the authority','unresolved whether the downstream transfer','does not determine whether the authority chain'])]
        initial.label=UNRESOLVED
        resolved=legal_story(random.Random(s),chain_depth=chain_depth,plaintiff_right='yes',authority='unknown',serious=True,distractors=6,late_reveal=True,forced_reveal=resolution)
        out.append((initial,resolved))
    return out

TOKEN_RE=re.compile(r"[A-Za-z0-9']+|[.,;:!?-]")
class Vocabulary:
    def __init__(self,stories):
        toks=set()
        for s in stories:
            for f in s.facts: toks.update(self.tokenize(f))
        self.itos=['<PAD>','<UNK>','<CLS>','<SEP>']+sorted(toks); self.stoi={t:i for i,t in enumerate(self.itos)}
    @staticmethod
    def tokenize(text): return [x.lower() for x in TOKEN_RE.findall(text)]
    def encode_fact(self,fact,max_words):
        ids=[self.stoi.get(t,1) for t in self.tokenize(fact)][:max_words]; return ids+[0]*(max_words-len(ids))
    def encode_flat(self,story,max_tokens):
        seq=[2]
        for f in story.facts:
            seq += [self.stoi.get(t,1) for t in self.tokenize(f)] + [3]
        seq=seq[:max_tokens]; return seq+[0]*(max_tokens-len(seq))
