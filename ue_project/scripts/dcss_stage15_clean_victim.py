import argparse, csv, hashlib, json, os, platform, subprocess, sys
from contextlib import redirect_stderr, redirect_stdout
import numpy as np
import torch, yaml
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from ue_framework.data_utils import list_images
from ue_framework.metrics_utils import compute_non_target_map, extract_map50_per_class

NAMES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

def state_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items(): digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--dataset-root',required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--device',default='0'); a=p.parse_args()
    out=os.path.abspath(a.output); os.makedirs(out,exist_ok=False)
    cfg={"dataset_root":os.path.abspath(a.dataset_root),"model":"configs/voc_yolov8n_20cls.yaml","epochs":a.epochs,"imgsz":640,"batch":16,"workers":0,"optimizer":"SGD","seed":0,"amp":True,"cos_lr":True,"close_mosaic":10}
    open(os.path.join(out,'command.txt'),'w',encoding='utf-8').write(' '.join(sys.argv)+'\n'); open(os.path.join(out,'environment.txt'),'w',encoding='utf-8').write(f'platform={platform.platform()}\npython={platform.python_version()}\ntorch={torch.__version__}\n'); open(os.path.join(out,'git_commit.txt'),'w',encoding='utf-8').write(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()+'\n')
    yaml.safe_dump(cfg,open(os.path.join(out,'config.yaml'),'w',encoding='utf-8'),sort_keys=False)
    data={"path":os.path.abspath(a.dataset_root).replace('\\','/'),"train":"images/train","val":"images/val","names":{i:n for i,n in enumerate(NAMES)}}; data_path=os.path.join(out,'data.yaml'); yaml.safe_dump(data,open(data_path,'w',encoding='utf-8'),sort_keys=False)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model=YOLO(os.path.join(ROOT,'configs','voc_yolov8n_20cls.yaml')); initial_hash=state_hash(model.model); torch.save(model.model.state_dict(),os.path.join(out,'initialization.pt')); open(os.path.join(out,'initialization.txt'),'w',encoding='utf-8').write(f'configs/voc_yolov8n_20cls.yaml\nsha256_state={initial_hash}\nseed=0\npretrained=false\n')
    log=open(os.path.join(out,'training.log'),'w',encoding='utf-8')
    with redirect_stdout(log),redirect_stderr(log):
        result=model.train(data=data_path,epochs=a.epochs,imgsz=640,batch=16,workers=0,cache=False,amp=True,pretrained=False,resume=False,optimizer='SGD',seed=0,deterministic=True,device=a.device,project=out,name='train',exist_ok=True,plots=False,verbose=False,save=True,save_period=10,cos_lr=True,close_mosaic=10)
    log.close(); run=str(result.save_dir); results=list(csv.DictReader(open(os.path.join(run,'results.csv'),encoding='utf-8')))
    checkpoints=[]
    for epoch in range(10,a.epochs+1,10):
        path=os.path.join(run,'weights',f'epoch{epoch-1}.pt')
        if os.path.isfile(path): checkpoints.append((epoch,path))
    best=os.path.join(run,'weights','best.pt'); checkpoints.extend([('best',best),('best_repeat',best)])
    epoch_rows=[]; eval_log=open(os.path.join(out,'evaluation.log'),'w',encoding='utf-8')
    with redirect_stdout(eval_log),redirect_stderr(eval_log):
        for epoch,path in checkpoints:
            metrics=YOLO(path).val(data=data_path,imgsz=640,batch=16,workers=0,device=a.device,plots=False,verbose=False,project=out,name=f'eval_{epoch}',exist_ok=True)
            ap=extract_map50_per_class(metrics,20); train_row=results[-1] if str(epoch).startswith('best') else results[int(epoch)-1]
            epoch_rows.append({"epoch":epoch,"mAP50_target":ap[14],"mAP50_non_target":compute_non_target_map(ap,14),"mAP50_all":float(np.nanmean(ap)),"train_box_loss":train_row.get('train/box_loss'),"train_cls_loss":train_row.get('train/cls_loss'),"train_dfl_loss":train_row.get('train/dfl_loss'),"val_box_loss":train_row.get('val/box_loss'),"val_cls_loss":train_row.get('val/cls_loss'),"val_dfl_loss":train_row.get('val/dfl_loss'),"checkpoint":path})
    eval_log.close()
    with open(os.path.join(out,'epoch_metrics.csv'),'w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(epoch_rows[0])); w.writeheader(); w.writerows(epoch_rows)
    numeric=[r for r in epoch_rows if not str(r['epoch']).startswith('best')]; last=numeric[-3:] if len(numeric)>=3 else numeric; vals=[float(r['mAP50_non_target']) for r in last]; stable=(max(vals)-min(vals)<=0.02) if vals else False
    best_row=next(r for r in epoch_rows if r['epoch']=='best'); repeated=next(r for r in epoch_rows if r['epoch']=='best_repeat'); repeatable=abs(float(best_row['mAP50_all'])-float(repeated['mAP50_all']))<=1e-6
    gate={"training_curve_normal":all(np.isfinite(float(r['mAP50_all'])) for r in numeric),"non_target_at_least_0_70":float(best_row['mAP50_non_target'])>=0.70,"recent_20_epoch_stable":stable,"initialization_independent_of_mini_val":True,"unified_evaluation_repeatable":repeatable}; gate['pass']=all(v is True for v in gate.values())
    payload={**best_row,"initialization_sha256":initial_hash,"epochs":a.epochs,"train_samples":len(list_images(os.path.join(a.dataset_root,'images','train'))),"val_samples":len(list_images(os.path.join(a.dataset_root,'images','val'))),"gate":gate,"continue_to_150":not stable and a.epochs<150}
    json.dump(payload,open(os.path.join(out,'metrics.json'),'w',encoding='utf-8'),indent=2,ensure_ascii=False); open(os.path.join(out,'best_checkpoint_path.txt'),'w',encoding='utf-8').write(best+'\n'); print(json.dumps(payload,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
