"""Offline descriptive/paired analysis for the registered SFT augmentation run."""
from __future__ import annotations
import collections,json,math,pathlib,statistics
import numpy as np
from scipy.stats import binomtest
from utils.run_equation_augmentation import RUN,read,rows,write,write_rows,sha

def wilson(k,n):
 if not n:return None
 z=1.95996398454;d=1+z*z/n;p=k/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-r,c+r]

def metrics(rs):
 n=len(rs);good=[r for r in rs if r['correct'] and not r['truncated']];lens=[len(r['completion_token_ids']) for r in rs];gl=[len(r['completion_token_ids']) for r in good]
 return {'n':n,'correct':len(good),'accuracy':len(good)/n if n else None,'wilson95':wilson(len(good),n),
  'full_trace_verified':sum(r['full_trace_verified'] for r in rs),'full_trace_verification_rate':sum(r['full_trace_verified'] for r in rs)/n if n else None,
  'unverified_trace':sum(not r['full_trace_verified'] for r in rs),'answer_extraction_success':sum(r['answer_extraction_success'] and not r['truncated'] for r in rs)/n if n else None,
  'truncation_rate':sum(r['truncated'] for r in rs)/n if n else None,'mean_completion_tokens':statistics.mean(lens) if lens else None,'median_completion_tokens':statistics.median(lens) if lens else None,
  'mean_correct_complete_tokens':statistics.mean(gl) if gl else None,'median_correct_complete_tokens':statistics.median(gl) if gl else None,'correct_complete_n':len(good),'total_output_tokens':sum(lens)}

def paired(a,b):
 # Positive differences favor second model for accuracy, but indicate longer
 # outputs for length. Same public question IDs in every pair.
 aa={r['id']:r for r in a};bb={r['id']:r for r in b}
 if set(aa)!=set(bb):raise ValueError('Paired panels differ')
 ids=sorted(aa);n=len(ids);rng=np.random.default_rng(20261116)
 def boot(v):
  if not len(v):return None
  v=np.asarray(v,dtype=float);means=v[rng.integers(0,len(v),size=(10000,len(v)))].mean(axis=1)
  return [float(x) for x in np.quantile(means,[.025,.975])]
 result={'n':n,'bootstrap_resamples':10000,'difference_direction':'second_minus_first'}
 for field in ('correct','truncated','full_trace_verified'):
  win=sum(not aa[i][field] and bb[i][field] for i in ids);loss=sum(aa[i][field] and not bb[i][field] for i in ids)
  v=[int(bb[i][field])-int(aa[i][field]) for i in ids]
  result[field]={'first_false_second_true':win,'first_true_second_false':loss,'mean_paired_difference':float(np.mean(v)) if n else None,
    'exact_mcnemar_p':float(binomtest(min(win,loss),win+loss,.5).pvalue) if win+loss else 1.,'paired_bootstrap95':boot(v)}
 differences=[len(bb[i]['completion_token_ids'])-len(aa[i]['completion_token_ids']) for i in ids]
 common=[i for i in ids if aa[i]['correct'] and bb[i]['correct'] and not aa[i]['truncated'] and not bb[i]['truncated']]
 dl=[len(bb[i]['completion_token_ids'])-len(aa[i]['completion_token_ids']) for i in common]
 result['all_length']={'mean_paired_difference':float(np.mean(differences)) if n else None,'paired_bootstrap95':boot(differences)}
 result['common_correct_complete_length']={'n':len(common),'mean_paired_difference':float(np.mean(dl)) if dl else None,'paired_bootstrap95':boot(dl)}
 return result

def run():
 results={a:rows(RUN/f'sft_{a}_validation/rows.jsonl') for a in 'ABC'}
 for a in results:
  if read(RUN/f'sft_{a}_validation/result.json')['status']!='completed':raise ValueError('Incomplete eval')
 summary={a:{t:metrics([r for r in rs if r['task_type']==t]) for t in ('numeral_system','gravity','equation_numeric')} for a,rs in results.items()}
 for a in results:
  summary[a]['macro_accuracy']=statistics.mean(summary[a][t]['accuracy'] for t in ('numeral_system','gravity','equation_numeric'))
  summary[a]['equation_subtypes']={sub:metrics([r for r in results[a] if r['task_type']=='equation_numeric' and r['deduce_or_guess']==sub]) for sub in ('deduce','guess')}
 comparisons={}
 for first,second in [('A','B'),('A','C'),('C','B')]:
  comparisons[first+'_vs_'+second]={t:paired([r for r in results[first] if r['task_type']==t],[r for r in results[second] if r['task_type']==t]) for t in ('numeral_system','gravity','equation_numeric')}
 analysis={'status':'completed','metrics':summary,'paired_comparisons':comparisons,'primary_comparison':'C_vs_B','holdout_used':False,
  'inference':'Exploratory paired comparisons on 50 Validation questions/task; no noninferiority claim from nonsignificance. Unadjusted p-values; multiple endpoints require caution.',
  'trace_scope':'Exact canonical Full Trace certificate only; free wording marked unverified, not mathematically incorrect. This conservative coverage cannot establish general reasoning quality.'}
 write(RUN/'sft_statistical_analysis.json',analysis)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4))
 for arm,label in [('B','Synthetic'),('C','Real Train control')]:
  curve=rows(RUN/f'sft_{arm}_train/training_curve.jsonl');ax.plot([x['optimizer_step'] for x in curve],[x['token_mean_loss'] for x in curve],label=label)
 ax.set(xlabel='Optimizer step',ylabel='Supervised token mean loss');ax.legend();fig.tight_layout();fig.savefig(RUN/'sft_training_curves.png',dpi=180);plt.close(fig)
 return analysis
if __name__=='__main__':run()
