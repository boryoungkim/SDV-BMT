#!/usr/bin/env python3
"""Report native-porting blockers without loading an SDK library or opening NPU."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[2]
def audit():
 result={'status':'WORKSPACE_READY_NATIVE_MODELS_NOT_YET_QUALIFIED','npu_opened':False,'models':{},'blockers':[]}
 for v in ['r18','r50','r101']:
  path=ROOT/f'configs/workloads/deeplabv3plus-{v}-cityscapes-v1-eagle-n.yaml'
  p=yaml.safe_load(path.read_text())['options']['device_program'];q=Path(p['qualification'])
  result['models'][v]={'plan_exists':Path(p['plan']).is_file(),'weights_exists':Path(p['weights']).is_file(),'native_qualification_exists':q.is_file(),'native_model_correctness':'NOT_RUN' if not q.exists() else 'RECEIPT_REQUIRES_REVALIDATION'}
  if not q.exists():result['blockers'].append(v+': native operator/module/stage/full gates pending; Python standalone PASS is not a C++ receipt')
 dataset=ROOT/'datasets/cityscapes-val500/index.jsonl';result['canonical_val500_present']=dataset.exists()
 if not dataset.exists():result['blockers'].append('Cityscapes val500 must be supplied/rebuilt under the original input digest')
 result['blockers']+=['L4 NPU golden not captured; baseline files are .yaml.in templates','Review Plan C/mixed precision/host output Cast and timing declarations before acceptance','Verify trace/lifetime/no-hot-path-allocation and staging physical-memory budget before full capture']
 return result
if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
