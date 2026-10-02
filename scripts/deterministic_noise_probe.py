"""Bounded baseline equivalence harness; does not affect official optimization settings."""
import argparse
import os
from pathlib import Path
import random
import runpy
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--implementation',required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
source=Path(args.implementation).resolve()
random.seed(0);np.random.seed(0);torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
sys.path.insert(0,str(source));sys.path.insert(0,str(source/'scripts'))
sys.argv=['train_noise_once.py','--config',str(source/'ue_framework/configs/exp_voc_person_legacy_kproto_ret_b2.yaml'),
          '--output',args.output,'--seed','0','--smoke-max-images','16','--smoke-max-steps','1']
runpy.run_path(str(source/'scripts/train_noise_once.py'),run_name='__main__')
