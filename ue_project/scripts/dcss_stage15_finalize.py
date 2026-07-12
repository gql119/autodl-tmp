import argparse,csv,json,math,os
import numpy as np

def load(path): return json.load(open(path,encoding='utf-8'))
def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r)); f=open(path,'w',newline='',encoding='utf-8'); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows); f.close()
def equal(a):
    rows=[load(os.path.join(a.root,f'diagnostic_{name}','metrics.json')) for name in a.ids]; by={r['experiment_id'].split('_')[0]:r for r in rows}; m0,m3=by['M0'],by['M3']
    checks={'target_energy':m3['target_projected_energy']>=m0['target_projected_energy'],'R_shift':m3['R_shift']>=2,'in_subspace':m3['target_in_subspace_ratio']>m0['target_in_subspace_ratio'],'leakage':m3['non_target_leakage']<=1.1*m0['non_target_leakage'],'coverage':m3['target_unit_coverage']>=.5,'finite':m3['finite']}
    write_csv(os.path.join(a.root,'equal_budget_summary.csv'),rows); json.dump({'checks':checks,'pass':all(checks.values()),'M0':m0,'M3':m3},open(os.path.join(a.root,'equal_budget_gate.json'),'w',encoding='utf-8'),indent=2)
    print(json.dumps({'checks':checks,'pass':all(checks.values())},indent=2))
def gradient(a):
    rows=list(csv.DictReader(open(a.csv,newline='',encoding='utf-8'))); number=len(rows)
    def ratio(key): return float(np.mean([float(r.get(key,'nan'))<0 for r in rows if r.get(key,'') not in ('','nan')]))
    class_keys=sorted({k for r in rows for k in r if k.startswith('cos_energy_class_')}); class_rows=[]
    for key in class_keys:
        vals=[float(r[key]) for r in rows if r.get(key,'') not in ('','nan')]; class_rows.append({'class_id':int(key.rsplit('_',1)[1]),'conflict_ratio':float(np.mean(np.array(vals)<0)),'mean_negative_cosine':float(np.mean([v for v in vals if v<0])) if any(v<0 for v in vals) else 0,'worst_negative_cosine':min(vals)})
    summary={'batch_count':number,'energy_leak_conflict_ratio':ratio('cos_energy_leak'),'energy_logit_conflict_ratio':ratio('cos_energy_logit'),'feasible_direction_ratio':float(np.mean([int(r['feasible_direction']) for r in rows])),'optimizer_type':rows[0]['update_mode'],'significant_conflict':ratio('cos_energy_leak')>=.25 or any(r['conflict_ratio']>=.25 for r in class_rows)}
    write_csv(os.path.join(a.output,'batch_gradient_metrics.csv'),rows); write_csv(os.path.join(a.output,'classwise_gradient_conflicts.csv'),class_rows); json.dump(summary,open(os.path.join(a.output,'gradient_summary.json'),'w',encoding='utf-8'),indent=2); print(json.dumps(summary,indent=2))
def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True); e=s.add_parser('equal');e.add_argument('--root',required=True);e.add_argument('--ids',nargs=4,required=True);e.set_defaults(fn=equal);g=s.add_parser('gradient');g.add_argument('--csv',required=True);g.add_argument('--output',required=True);g.set_defaults(fn=gradient); a=p.parse_args(); os.makedirs(getattr(a,'output',getattr(a,'root','.')),exist_ok=True);a.fn(a)
if __name__=='__main__':main()
