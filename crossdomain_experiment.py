from __future__ import annotations
from pathlib import Path
import csv,json,random
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crossdomain_data import *
from crossdomain_models import *

BUDGETS=[16,32,64,128,256]

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def plot_learning(path,rows):
    fig=plt.figure(figsize=(9.5,5.7)); ax=fig.add_subplot(111); x=[r['legal_examples'] for r in rows]
    for k,l in [('transfer_accuracy','Toddler → legal transfer (frozen operator)'),('scratch_accuracy','Legal relational model from scratch'),('scrambled_accuracy','Scrambled-operator transfer control'),('transformer_accuracy','Tiny transformer')]: ax.plot(x,[r[k] for r in rows],marker='o',label=l)
    ax.axhline(1/3,ls='--',label='3-class chance'); ax.set_xscale('log',base=2); ax.set_xticks(x,labels=[str(v) for v in x]); ax.set_ylim(.25,1.02); ax.set_xlabel('Labeled legal examples'); ax.set_ylabel('Held-out hard legal accuracy'); ax.set_title('Experiment 4: Cross-Domain Sample Efficiency'); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)

def plot_special(path,metrics):
    fig=plt.figure(figsize=(9.2,5.5)); ax=fig.add_subplot(111); models=['transfer','scratch','scrambled','transformer']; x=np.arange(4); w=.36
    ax.bar(x-w/2,[metrics[m]['material_flip_correct'] for m in models],w,label='Material one-fact flip correct'); ax.bar(x+w/2,[metrics[m]['irrelevant_invariance'] for m in models],w,label='Irrelevant changes preserve answer'); ax.set_xticks(x,['Transfer','Scratch','Scrambled','Transformer']); ax.set_ylim(0,1.03); ax.set_ylabel('Fraction'); ax.set_title('Experiment 4: Material Sensitivity vs Surface Invariance'); ax.grid(True,axis='y',alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)

def plot_revision(path,metrics):
    fig=plt.figure(figsize=(8.8,5.4)); ax=fig.add_subplot(111); models=['transfer','scratch','scrambled','transformer']; ax.bar(np.arange(4),[metrics[m]['revelation_revision_correct'] for m in models]); ax.set_xticks(np.arange(4),['Transfer','Scratch','Scrambled','Transformer']); ax.set_ylim(0,1.03); ax.set_ylabel('Correct initial uncertainty → correct revised judgment'); ax.set_title('Experiment 4: Delayed Relational Revelation'); ax.grid(True,axis='y',alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)

def pair_metrics_rel(model,vocab,material,irrelevant,revelation):
    pm,_=predict_relational(model,[s for p in material for s in p],vocab); pm=pm.reshape(-1,2); mi=np.mean([(pa==a.label and pb==b.label and pa!=pb) for (a,b),(pa,pb) in zip(material,pm)])
    pi,_=predict_relational(model,[s for p in irrelevant for s in p],vocab); pi=pi.reshape(-1,2); inv=np.mean([(pa==a.label and pb==b.label and pa==pb) for (a,b),(pa,pb) in zip(irrelevant,pi)])
    pr,_=predict_relational(model,[s for p in revelation for s in p],vocab); pr=pr.reshape(-1,2); rev=np.mean([(pa==UNRESOLVED and pb==b.label) for (a,b),(pa,pb) in zip(revelation,pr)])
    return {'material_flip_correct':float(mi),'irrelevant_invariance':float(inv),'revelation_revision_correct':float(rev)}

def pair_metrics_tf(model,vocab,material,irrelevant,revelation):
    pm,_=predict_transformer(model,[s for p in material for s in p],vocab); pm=pm.reshape(-1,2); mi=np.mean([(pa==a.label and pb==b.label and pa!=pb) for (a,b),(pa,pb) in zip(material,pm)])
    pi,_=predict_transformer(model,[s for p in irrelevant for s in p],vocab); pi=pi.reshape(-1,2); inv=np.mean([(pa==a.label and pb==b.label and pa==pb) for (a,b),(pa,pb) in zip(irrelevant,pi)])
    pr,_=predict_transformer(model,[s for p in revelation for s in p],vocab); pr=pr.reshape(-1,2); rev=np.mean([(pa==UNRESOLVED and pb==b.label) for (a,b),(pa,pb) in zip(revelation,pr)])
    return {'material_flip_correct':float(mi),'irrelevant_invariance':float(inv),'revelation_revision_correct':float(rev)}

def locked_criteria(rows,special):
    d={r['legal_examples']:r for r in rows}; b64=d[64]; b128=d[128]
    c={'few_shot_advantage_vs_scratch_at_64':b64['transfer_accuracy']-b64['scratch_accuracy']>=.15,'few_shot_advantage_vs_scrambled_at_64':b64['transfer_accuracy']-b64['scrambled_accuracy']>=.15,'few_shot_advantage_vs_transformer_at_64':b64['transfer_accuracy']-b64['transformer_accuracy']>=.10,'absolute_hard_legal_accuracy_at_128':b128['transfer_accuracy']>=.85,'material_flip_at_256':special['transfer']['material_flip_correct']>=.85,'irrelevant_invariance_at_256':special['transfer']['irrelevant_invariance']>=.90,'delayed_revelation_revision_at_256':special['transfer']['revelation_revision_correct']>=.80}; return c,all(c.values())

def run_crossdomain_experiment(output_dir='crossdomain_results',seed=41,toddler_steps=900,legal_steps=325,batch_size=64,budgets=None,event_callback=None,smoke=False,cleanup=False,num_states=32):
    """cleanup=False reproduces the original continuous-state run. cleanup=True snaps the
    relational state onto a learned finite set after every composition (Experiments 1-3 closure)."""
    if budgets is None: budgets=BUDGETS[:]
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    ttrain=generate_toddler_dataset(seed+1,800 if smoke else 5000,.35); teval=generate_toddler_dataset(seed+2,240 if smoke else 1200,.50); tvocab=Vocabulary(ttrain+teval); base=RelationalCaseModel(len(tvocab.itos),cleanup=cleanup,num_states=num_states)
    def tprog(step,total,loss,acc):
        if event_callback: event_callback('toddler_train',{'step':step,'total':total,'loss':loss,'acc':acc,'overall_pct':.18*step/total})
    train_relational(base,ttrain,tvocab,toddler_steps,batch_size,2.5e-3,seed+3,tprog); pt,yt=predict_relational(base,teval,tvocab); tacc=accuracy(pt,yt)
    if event_callback: event_callback('status',{'text':f'Toddler pretraining finished: held-out accuracy {tacc:.3f}'})
    pool=generate_legal_dataset(seed+10,max(budgets),2,1,.25,(3,6)); hard=generate_legal_dataset(seed+11,240 if smoke else 1200,4,4,.55,(6,9)); material=legal_material_pairs(seed+12,40 if smoke else 240); irrelevant=legal_irrelevant_pairs(seed+13,40 if smoke else 240); revelation=legal_revelation_pairs(seed+14,40 if smoke else 240)
    lvocab=Vocabulary(pool+hard+[s for p in material for s in p]+[s for p in irrelevant for s in p]+[s for p in revelation for s in p])
    rows=[]; total=len(budgets)*4; done=0; max_models={}
    for budget in budgets:
        subset=pool[:budget]
        transfer=base.transfer_shell(len(lvocab.itos))
        def prog(name):
            def cb(step,tot,loss,acc):
                if event_callback and (step==1 or step%50==0 or step==tot): event_callback('legal_train',{'budget':budget,'model':name,'step':step,'total':tot,'loss':loss,'acc':acc,'overall_pct':.18+.82*((done+step/tot)/total)})
            return cb
        train_relational(transfer,subset,lvocab,legal_steps,min(batch_size,max(16,budget)),3e-3,seed+100+budget,prog('transfer')); done+=1
        scratch=RelationalCaseModel(len(lvocab.itos),cleanup=cleanup,num_states=num_states); train_relational(scratch,subset,lvocab,legal_steps,min(batch_size,max(16,budget)),2.5e-3,seed+200+budget,prog('scratch')); done+=1
        scrambled=base.scrambled_transfer_shell(len(lvocab.itos),seed+300+budget); train_relational(scrambled,subset,lvocab,legal_steps,min(batch_size,max(16,budget)),3e-3,seed+400+budget,prog('scrambled')); done+=1
        transformer=TinyStoryTransformer(len(lvocab.itos)); train_transformer(transformer,subset,lvocab,legal_steps,min(batch_size,max(16,budget)),2e-3,seed+500+budget,prog('transformer')); done+=1
        p1,y=predict_relational(transfer,hard,lvocab); p2,_=predict_relational(scratch,hard,lvocab); p3,_=predict_relational(scrambled,hard,lvocab); p4,_=predict_transformer(transformer,hard,lvocab)
        row={'legal_examples':budget,'transfer_accuracy':accuracy(p1,y),'scratch_accuracy':accuracy(p2,y),'scrambled_accuracy':accuracy(p3,y),'transformer_accuracy':accuracy(p4,y)}; rows.append(row)
        if budget==max(budgets): max_models={'transfer':transfer,'scratch':scratch,'scrambled':scrambled,'transformer':transformer,'truth':y,'preds':{'transfer':p1,'scratch':p2,'scrambled':p3,'transformer':p4}}
        if event_callback: event_callback('budget_done',{'budget':budget,'row':row,'overall_pct':.18+.82*done/total})
    special={'transfer':pair_metrics_rel(max_models['transfer'],lvocab,material,irrelevant,revelation),'scratch':pair_metrics_rel(max_models['scratch'],lvocab,material,irrelevant,revelation),'scrambled':pair_metrics_rel(max_models['scrambled'],lvocab,material,irrelevant,revelation),'transformer':pair_metrics_tf(max_models['transformer'],lvocab,material,irrelevant,revelation)}
    criteria,allpass=locked_criteria(rows,special) if 64 in budgets and 128 in budgets and max(budgets)>=256 else ({},False)
    write_csv(out/'crossdomain_sample_efficiency.csv',rows); plot_learning(out/'crossdomain_sample_efficiency.png',rows); plot_special(out/'crossdomain_material_vs_irrelevant.png',special); plot_revision(out/'crossdomain_revelation_revision.png',special)
    conf={m:confusion(max_models['preds'][m],max_models['truth']) for m in ['transfer','scratch','scrambled','transformer']}
    report={'experiment':'Experiment 4 — Cross-Domain Coherence Transfer','seed':seed,'locked_before_run':True,'state_cleanup':cleanup,'num_relation_states':(num_states if cleanup else None),'core_hypothesis':'A relational state-update principle learned from simple toddler moral provenance can provide measurable few-shot leverage in a semantically and institutionally different conversion-style legal domain when the operator is frozen and only the new domain encoder/head may adapt.','domains':{'A':'Toddler toy-taking / taking-back / permission / provenance','B':'Synthetic Arizona-style conversion cases with custody chains, authority, immediate possessory right, and serious interference'},'anti_cheating_design':['Separate domain vocabularies and fresh legal encoder/head.','Only relational operator and initial state geometry transfer.','Transferred operator frozen during legal training.','Legal training chain depth 1-2; hard test depth 4.','Scratch relational control gets identical legal labels.','Scrambled-operator control keeps weight scale but destroys learned recurrence.','Tiny transformer gets identical legal labels.','Material flips, irrelevant perturbations, and delayed revelation scored separately.'],'legal_ground_truth_scope':'Narrow controlled conversion-style core, not a legal expert system: immediate right to possession, intentional dominion/control, serious interference, and operative authorization.','training':{'toddler_steps':toddler_steps,'legal_steps_per_model_per_budget':legal_steps,'batch_size':batch_size,'legal_budgets':budgets,'toddler_train_examples':len(ttrain),'toddler_eval_examples':len(teval),'hard_legal_test_examples':len(hard)},'toddler_heldout_accuracy':tacc,'sample_efficiency':rows,'special_tests_at_max_budget':special,'locked_success_criteria':{'at_64_examples_transfer_minus_scratch':'>= 0.15','at_64_examples_transfer_minus_scrambled':'>= 0.15','at_64_examples_transfer_minus_transformer':'>= 0.10','at_128_examples_transfer_hard_legal_accuracy':'>= 0.85','material_one_fact_flip_at_256':'>= 0.85','irrelevant_surface_invariance_at_256':'>= 0.90','delayed_revelation_revision_at_256':'>= 0.80'},'criterion_results':criteria,'locked_all_criteria_pass':allpass,'parameter_counts_at_max_budget':{'transfer_total':parameter_count(max_models['transfer']),'transfer_trainable_legal_adapter':parameter_count(max_models['transfer'],True),'scratch_total_and_trainable':parameter_count(max_models['scratch']),'scrambled_total':parameter_count(max_models['scrambled']),'transformer_total':parameter_count(max_models['transformer'])},'hard_test_confusion_matrices_rows_truth_cols_prediction':conf,'labels':{'toddler':TODDLER_LABELS,'legal':LEGAL_LABELS},'interpretation_rule':'Primary result is cross-domain sample-efficiency advantage, not merely final accuracy. Positive transfer requires intact toddler-pretrained frozen operator to beat scratch and scrambled controls while reacting to material provenance and ignoring irrelevant surface variation. Mixed results remain mixed.'}
    (out/'crossdomain_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['EXPERIMENT 4 SAMPLE CASES','=========================','','TODDLER EXAMPLES','-----------------']
    for s in teval[:5]: lines+=[' | '.join(s.facts),f'TARGET: {TODDLER_LABELS[s.label]}','']
    lines+=['','HARD LEGAL EXAMPLES','-------------------']
    for s in hard[:5]: lines+=[' | '.join(s.facts),f'TARGET: {LEGAL_LABELS[s.label]}','']
    (out/'sample_cases.txt').write_text('\n'.join(lines),encoding='utf-8')
    torch.save({'toddler_pretrained':base.state_dict(),'max_budget_transfer':max_models['transfer'].state_dict(),'max_budget_scratch':max_models['scratch'].state_dict(),'max_budget_scrambled':max_models['scrambled'].state_dict(),'max_budget_transformer':max_models['transformer'].state_dict(),'report':report},out/'crossdomain_models.pt')
    if event_callback: event_callback('done_status',{'text':'Experiment complete. All plots and report written.','overall_pct':1.0})
    return {'output_dir':str(out.resolve()),'report':report}

if __name__=='__main__':
    r=run_crossdomain_experiment(); print(r['output_dir']); print(json.dumps(r['report']['sample_efficiency'],indent=2))
