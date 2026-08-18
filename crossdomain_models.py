from __future__ import annotations
import copy, math, random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from crossdomain_data import Story, Vocabulary

COMMITMENT_WEIGHT=0.25   # weight on the codebook commitment term when cleanup is enabled

def cpu_setup(): torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
def parameter_count(model,trainable_only=False): return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))

def encode_rel_batch(stories,vocab,max_facts=18,max_words=26,device=torch.device('cpu')):
    B=len(stories); x=torch.zeros(B,max_facts,max_words,dtype=torch.long,device=device); mask=torch.zeros(B,max_facts,dtype=torch.bool,device=device); y=torch.tensor([s.label for s in stories],dtype=torch.long,device=device)
    for b,s in enumerate(stories):
        for i,f in enumerate(s.facts[:max_facts]): x[b,i]=torch.tensor(vocab.encode_fact(f,max_words),device=device); mask[b,i]=True
    return x,mask,y

def encode_tf_batch(stories,vocab,max_tokens=224,device=torch.device('cpu')):
    x=torch.tensor([vocab.encode_flat(s,max_tokens) for s in stories],dtype=torch.long,device=device); y=torch.tensor([s.label for s in stories],dtype=torch.long,device=device); return x,y

class FactEncoder(nn.Module):
    def __init__(self,vocab_size,token_dim=32,event_dim=32):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,token_dim,padding_idx=0); self.net=nn.Sequential(nn.Linear(token_dim,64),nn.GELU(),nn.Linear(64,event_dim),nn.Tanh())
    def forward(self,x):
        emb=self.embedding(x); m=(x!=0).unsqueeze(-1); pooled=(emb*m).sum(2)/m.sum(2).clamp_min(1); return self.net(pooled)

class RelationCodebook(nn.Module):
    """Discrete cleanup memory. Snaps the state onto a learned finite set after each
    composition, which is the closure property that made Experiments 1-3 hold up past
    their training depth. Straight-through: discrete forward, differentiable backward."""
    def __init__(self,state_dim,num_states=32):
        super().__init__(); self.num_states=num_states; self.states=nn.Parameter(F.normalize(torch.randn(num_states,state_dim),dim=-1))
    def forward(self,h):
        h_n=F.normalize(h,dim=-1); book=F.normalize(self.states,dim=-1); nearest=book[(h_n@book.t()).argmax(dim=-1)]
        commitment=F.mse_loss(h_n,nearest.detach())+F.mse_loss(nearest,h_n.detach())
        return h_n+(nearest-h_n).detach(),commitment

class RelationalOperator(nn.Module):
    def __init__(self,state_dim=40,event_dim=32,cleanup=False,num_states=32):
        super().__init__(); self.state_dim=state_dim; self.event_dim=event_dim; self.gate=nn.Sequential(nn.Linear(state_dim+event_dim,80),nn.GELU(),nn.Linear(80,state_dim*2))
        self.cleanup=cleanup; self.codebook=RelationCodebook(state_dim,num_states) if cleanup else None; self.last_commitment=torch.zeros(())
    def forward(self,state,event):
        h=self.gate(torch.cat([state,event],-1)); proposal,g=h.chunk(2,-1); proposal=torch.tanh(proposal); g=torch.sigmoid(g)
        out=F.normalize(g*proposal+(1-g)*state,dim=-1)
        if self.codebook is not None: out,self.last_commitment=self.codebook(out)
        return out

class RelationalCaseModel(nn.Module):
    def __init__(self,vocab_size,token_dim=32,event_dim=32,state_dim=40,num_classes=3,cleanup=False,num_states=32):
        super().__init__(); self.encoder=FactEncoder(vocab_size,token_dim,event_dim); self.operator=RelationalOperator(state_dim,event_dim,cleanup=cleanup,num_states=num_states); self.initial_state=nn.Parameter(F.normalize(torch.randn(state_dim),dim=0)); self.head=nn.Sequential(nn.Linear(state_dim,48),nn.GELU(),nn.Linear(48,num_classes))
    def forward(self,x,mask):
        events=self.encoder(x); B,FN,_=events.shape; state=F.normalize(self.initial_state,dim=0).unsqueeze(0).expand(B,-1)
        for i in range(FN):
            active=mask[:,i]
            if not active.any(): break
            nxt=self.operator(state,events[:,i]); state=torch.where(active.unsqueeze(-1),nxt,state)
        return self.head(state)
    def transfer_shell(self,new_vocab_size):
        m=RelationalCaseModel(new_vocab_size,token_dim=self.encoder.embedding.embedding_dim,event_dim=self.operator.event_dim,state_dim=self.operator.state_dim,num_classes=self.head[-1].out_features,cleanup=self.operator.cleanup,num_states=(self.operator.codebook.num_states if self.operator.codebook is not None else 32)); m.operator.load_state_dict(copy.deepcopy(self.operator.state_dict())); m.initial_state.data.copy_(self.initial_state.data)
        for p in m.operator.parameters(): p.requires_grad=False
        m.initial_state.requires_grad=False
        return m
    def scrambled_transfer_shell(self,new_vocab_size,seed):
        m=self.transfer_shell(new_vocab_size); rng=random.Random(seed); perm=list(range(m.operator.state_dim)); rng.shuffle(perm); perm=torch.tensor(perm,dtype=torch.long)
        last=m.operator.gate[-1]
        with torch.no_grad():
            W,b=last.weight.clone(),last.bias.clone(); d=m.operator.state_dim; idx=torch.cat([perm,perm+d]); last.weight.copy_(W[idx]); last.bias.copy_(b[idx])
        return m

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=224):
        super().__init__(); pe=torch.zeros(max_len,d_model); pos=torch.arange(max_len).float().unsqueeze(1); div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); self.register_buffer('pe',pe)
    def forward(self,x): return x+self.pe[:x.size(1)].unsqueeze(0)

class TinyStoryTransformer(nn.Module):
    def __init__(self,vocab_size,max_tokens=224,d_model=48):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,d_model,padding_idx=0); self.pos=PositionalEncoding(d_model,max_tokens); layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=4,dim_feedforward=96,dropout=0.0,batch_first=True); self.encoder=nn.TransformerEncoder(layer,num_layers=2); self.head=nn.Linear(d_model,3)
    def forward(self,tokens):
        pad=tokens==0; x=self.pos(self.embedding(tokens)); x=self.encoder(x,src_key_padding_mask=pad); return self.head(x[:,0])

def _sample(ds,batch,rng): return [ds[rng.randrange(len(ds))] for _ in range(batch)]

def train_relational(model,ds,vocab,steps,batch_size,lr,seed,progress=None):
    cpu_setup(); dev=torch.device('cpu'); model.to(dev).train(); pars=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(pars,lr=lr,weight_decay=1e-4); rng=random.Random(seed); ce=nn.CrossEntropyLoss()
    for step in range(1,steps+1):
        b=_sample(ds,batch_size,rng); x,m,y=encode_rel_batch(b,vocab,device=dev); opt.zero_grad(set_to_none=True); logits=model(x,m); loss=ce(logits,y)
        if model.operator.codebook is not None: loss=loss+COMMITMENT_WEIGHT*model.operator.last_commitment
        loss.backward(); torch.nn.utils.clip_grad_norm_(pars,1.0); opt.step()
        if progress and (step==1 or step%25==0 or step==steps): progress(step,steps,float(loss.detach()),float((logits.argmax(1)==y).float().mean()))
    return model

def train_transformer(model,ds,vocab,steps,batch_size,lr,seed,progress=None):
    cpu_setup(); dev=torch.device('cpu'); model.to(dev).train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); rng=random.Random(seed); ce=nn.CrossEntropyLoss()
    for step in range(1,steps+1):
        b=_sample(ds,batch_size,rng); x,y=encode_tf_batch(b,vocab,device=dev); opt.zero_grad(set_to_none=True); logits=model(x); loss=ce(logits,y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if progress and (step==1 or step%25==0 or step==steps): progress(step,steps,float(loss.detach()),float((logits.argmax(1)==y).float().mean()))
    return model

@torch.no_grad()
def predict_relational(model,ds,vocab,batch_size=128):
    model.eval(); ps=[]; ys=[]
    for i in range(0,len(ds),batch_size):
        b=ds[i:i+batch_size]; x,m,y=encode_rel_batch(b,vocab); ps+=model(x,m).argmax(1).cpu().tolist(); ys+=y.cpu().tolist()
    return np.array(ps),np.array(ys)
@torch.no_grad()
def predict_transformer(model,ds,vocab,batch_size=128):
    model.eval(); ps=[]; ys=[]
    for i in range(0,len(ds),batch_size):
        b=ds[i:i+batch_size]; x,y=encode_tf_batch(b,vocab); ps+=model(x).argmax(1).cpu().tolist(); ys+=y.cpu().tolist()
    return np.array(ps),np.array(ys)
def accuracy(p,y): return float((p==y).mean())
def confusion(p,y,n=3):
    m=np.zeros((n,n),dtype=int)
    for yy,pp in zip(y,p): m[int(yy),int(pp)]+=1
    return m.tolist()
