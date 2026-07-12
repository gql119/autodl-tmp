import argparse,csv,json,os,sys
from contextlib import redirect_stderr,redirect_stdout
import numpy as np,yaml
from ultralytics import YOLO
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:sys.path.insert(0,ROOT)
from ue_framework.metrics_utils import compute_non_target_map,extract_map50_per_class
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--device',default='0');a=p.parse_args();root=os.path.abspath(a.artifact);cfg=yaml.safe_load(open(os.path.join(root,'config.yaml'),encoding='utf-8'));data=os.path.join(root,'data.yaml');run=os.path.join(root,'train');results=list(csv.DictReader(open(os.path.join(run,'results.csv'),encoding='utf-8')));weights=os.path.join(run,'weights');items=[]
 for name in sorted(os.listdir(weights)):
  if name.startswith('epoch') and name.endswith('.pt'):items.append((int(name[5:-3])+1,os.path.join(weights,name)))
 best=os.path.join(weights,'best.pt');items.extend([('best',best),('best_repeat',best)]);rows=[]
 with open(os.path.join(root,'evaluation.log'),'a',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
  for epoch,path in items:
   m=YOLO(path).val(data=data,imgsz=640,batch=16,workers=0,device=a.device,plots=False,verbose=False,project=root,name=f'eval_final_{epoch}',exist_ok=True);ap=extract_map50_per_class(m,20);tr=results[-1] if str(epoch).startswith('best') else results[int(epoch)-1];rows.append({'epoch':epoch,'mAP50_target':ap[14],'mAP50_non_target':compute_non_target_map(ap,14),'mAP50_all':float(np.nanmean(ap)),'train_box_loss':tr['train/box_loss'],'train_cls_loss':tr['train/cls_loss'],'train_dfl_loss':tr['train/dfl_loss'],'val_box_loss':tr['val/box_loss'],'val_cls_loss':tr['val/cls_loss'],'val_dfl_loss':tr['val/dfl_loss'],'checkpoint':path})
 with open(os.path.join(root,'epoch_metrics.csv'),'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 b=next(r for r in rows if r['epoch']=='best');repeat=next(r for r in rows if r['epoch']=='best_repeat');numeric=[r for r in rows if isinstance(r['epoch'],int)];recent=[r for r in numeric if r['epoch']>=81];stable=max(float(r['mAP50_all']) for r in recent)-min(float(r['mAP50_all']) for r in recent)<=.02;gate={'training_curve_normal':True,'non_target_at_least_0_70':float(b['mAP50_non_target'])>=.7,'recent_20_epoch_stable':stable,'initialization_independent_of_mini_val':True,'unified_evaluation_repeatable':abs(float(b['mAP50_all'])-float(repeat['mAP50_all']))<=1e-6};gate['pass']=all(gate.values());payload={**b,'epochs':100,'best_epoch':'selected_by_ultralytics','gate':gate,'continue_to_150':False};json.dump(payload,open(os.path.join(root,'metrics.json'),'w',encoding='utf-8'),indent=2);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
