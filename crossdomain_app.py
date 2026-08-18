from __future__ import annotations
import json,queue,threading,tkinter as tk
from tkinter import ttk,messagebox
from crossdomain_experiment import run_crossdomain_experiment

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('Relational Token Lab — Experiment 4'); self.geometry('1180x830'); self.minsize(980,720); self.q=queue.Queue(); self.worker=None; self.build(); self.after(100,self.poll)
    def build(self):
        outer=ttk.Frame(self,padding=12); outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='Experiment 4 — Cross-Domain Coherence Transfer',font=('Segoe UI',18,'bold')).pack(anchor='w')
        ttk.Label(outer,text='Toddler moral provenance → frozen relational operator → few-shot conversion-style legal reasoning.').pack(anchor='w',pady=(0,10))
        box=ttk.LabelFrame(outer,text='Locked default run',padding=10); box.pack(fill='x')
        self.seed=tk.StringVar(value='41'); self.ts=tk.StringVar(value='900'); self.ls=tk.StringVar(value='325'); self.batch=tk.StringVar(value='64')
        for i,(lab,var) in enumerate([('Seed',self.seed),('Toddler pretrain steps',self.ts),('Legal steps / model / budget',self.ls),('Batch size',self.batch)]):
            ttk.Label(box,text=lab).grid(row=0,column=i,sticky='w',padx=6); ttk.Entry(box,textvariable=var,width=20).grid(row=1,column=i,sticky='we',padx=6); box.columnconfigure(i,weight=1)
        ttk.Label(box,text='Legal budgets locked at 16, 32, 64, 128, 256 labeled examples.').grid(row=2,column=0,columnspan=4,sticky='w',padx=6,pady=(6,0))
        self.cleanup=tk.BooleanVar(value=False)
        ttk.Checkbutton(box,text='Discrete state cleanup (snap to a learned finite state set after every step)',variable=self.cleanup).grid(row=3,column=0,columnspan=4,sticky='w',padx=6,pady=(6,0))
        ttk.Label(box,text='Off = original continuous-state run. On = closure property from Experiments 1-3.',foreground='#555555').grid(row=4,column=0,columnspan=4,sticky='w',padx=6)
        row=ttk.Frame(outer); row.pack(fill='x',pady=10); self.runbtn=ttk.Button(row,text='Run Experiment 4',command=self.start); self.runbtn.pack(side='left'); self.progress=ttk.Progressbar(row,mode='determinate',maximum=100); self.progress.pack(side='right',fill='x',expand=True,padx=(20,0))
        nb=ttk.Notebook(outer); nb.pack(fill='both',expand=True); a=ttk.Frame(nb,padding=8); b=ttk.Frame(nb,padding=8); c=ttk.Frame(nb,padding=8); nb.add(a,text='Run log'); nb.add(b,text='Locked design'); nb.add(c,text='Result summary')
        self.log=tk.Text(a,wrap='word',font=('Consolas',10)); self.log.pack(fill='both',expand=True)
        lock=tk.Text(b,wrap='word',font=('Segoe UI',10)); lock.pack(fill='both',expand=True); lock.insert('1.0',LOCK_TEXT); lock.configure(state='disabled')
        self.summary=tk.Text(c,wrap='word',font=('Consolas',10)); self.summary.pack(fill='both',expand=True); self.append('Ready. CPU-only. This run is heavier than Experiment 3.\n')
    def start(self):
        if self.worker and self.worker.is_alive(): return
        try:
            vals=(int(self.seed.get()),int(self.ts.get()),int(self.ls.get()),int(self.batch.get()),bool(self.cleanup.get()))
            if vals[1]<100 or vals[2]<50 or vals[3]<16: raise ValueError('Use at least 100 toddler steps, 50 legal steps, batch >=16.')
        except Exception as e: messagebox.showerror('Invalid settings',str(e)); return
        self.runbtn.configure(state='disabled'); self.progress['value']=0; self.log.delete('1.0','end'); self.summary.delete('1.0','end'); self.append('Starting Experiment 4...\n\n'); self.worker=threading.Thread(target=self.work,args=vals,daemon=True); self.worker.start()
    def work(self,seed,ts,ls,batch,cleanup):
        try:
            arm='cleanup_on' if cleanup else 'cleanup_off'
            r=run_crossdomain_experiment(output_dir=f'crossdomain_results_v2/gui_{arm}',seed=seed,toddler_steps=ts,legal_steps=ls,batch_size=batch,cleanup=cleanup,event_callback=lambda k,p:self.q.put((k,p))); self.q.put(('complete',r))
        except Exception as e: self.q.put(('error',repr(e)))
    def poll(self):
        try:
            while True:
                k,p=self.q.get_nowait()
                if isinstance(p,dict) and 'overall_pct' in p: self.progress['value']=100*p['overall_pct']
                if k=='toddler_train': self.append(f"Toddler {p['step']:4d}/{p['total']} | loss={p['loss']:.5f} | batch acc={p['acc']:.3f}\n")
                elif k=='legal_train': self.append(f"Legal n={p['budget']:3d} | {p['model']:11s} | {p['step']:3d}/{p['total']} | loss={p['loss']:.5f} | batch acc={p['acc']:.3f}\n")
                elif k=='budget_done':
                    r=p['row']; self.append(f"\nBudget {p['budget']} HARD TEST: transfer={r['transfer_accuracy']:.3f}, scratch={r['scratch_accuracy']:.3f}, scrambled={r['scrambled_accuracy']:.3f}, transformer={r['transformer_accuracy']:.3f}\n\n")
                elif k in ('status','done_status'): self.append('\n'+p['text']+'\n')
                elif k=='complete':
                    self.progress['value']=100; self.runbtn.configure(state='normal'); rep=p['report']; self.append('\nExperiment complete.\nResults folder:\n'+p['output_dir']+'\n'); compact={x:rep[x] for x in ['state_cleanup','num_relation_states','toddler_heldout_accuracy','sample_efficiency','special_tests_at_max_budget','criterion_results','locked_all_criteria_pass','parameter_counts_at_max_budget','interpretation_rule']}; self.summary.insert('1.0',json.dumps(compact,indent=2))
                elif k=='error': self.runbtn.configure(state='normal'); self.append('\nERROR:\n'+p+'\n'); messagebox.showerror('Experiment failed',p)
        except queue.Empty: pass
        self.after(100,self.poll)
    def append(self,t): self.log.insert('end',t); self.log.see('end')

LOCK_TEXT='''EXPERIMENT 4 IS LOCKED BEFORE THE RESULT\n\nQUESTION\nCan a relational state-update principle learned only from simple toddler toy-taking / taking-back / permission / provenance cases provide measurable leverage in a semantically different, institutionally complex legal domain?\n\nWHAT TRANSFERS\n  learned relational operator\n  learned initial state geometry\n\nWHAT DOES NOT TRANSFER\n  toddler vocabulary\n  toddler word embeddings\n  toddler decision head\n\nThe relational operator is FROZEN during legal learning.\n\nCOMPARATORS\n1. Transfer — toddler-pretrained frozen operator.\n2. Scratch — same relational architecture, legal only.\n3. Scrambled — same transferred weight scale, recurrence broken before freezing.\n4. Tiny transformer — same legal labels.\n\nDOMAIN SHIFT\nToddler: short moral/social provenance stories.\nLegal train: conversion-style authority/custody chains depth 1–2.\nHard legal test: depth 4, delayed material facts, more distractors.\n\nSPECIAL TESTS\nA. One material relational fact changes; answer must flip.\nB. Surface details change; answer must remain stable.\nC. Initial record is indeterminate; late material fact must trigger correct revision.\n\nLOCKED CRITERIA\nAt 64 legal examples:\n  Transfer - Scratch >= 15 points\n  Transfer - Scrambled >= 15 points\n  Transfer - Transformer >= 10 points\nAt 128 examples:\n  Transfer hard-test accuracy >= 85%\nAt 256 examples:\n  Material flip >= 85%\n  Irrelevant invariance >= 90%\n  Delayed revelation revision >= 80%\n\nA mixed result stays mixed. We do not crush it into a stupid boolean.\n\nLEGAL SCOPE\nThe legal generator is a controlled conversion-style experimental world, not a legal expert system.\n'''

if __name__=='__main__': App().mainloop()
