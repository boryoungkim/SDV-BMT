#!/usr/bin/env python3
"""Generate Eagle-N-only configs; leave canonical workload/input/measurement semantics unchanged."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sdv_bmt.dut.eagle_n.adapter import PRECISION_DECLARATION  # noqa: E402
# Host memory budgets (INTERNAL guards, not guide values), ONE value for every supported plan.
# Every supported plan has BF16 activations (precision bf16 or bfp8w; FP32-conv plans were removed 2026-11-18), so each
# staged item is the same 12 MiB BF16 input. Sized from the R18 RSS probe (l1_20261115T052755_396307Z: base ~305 MiB,
# 11.63 MiB per staged item; 500 items ~6.0 GiB) -> staging 6.5 GiB, process RSS 7 GiB; confirmed for R50 bf16
# (10.5 MiB/item, evidence log §48) and used by R50/R101 bf16 L3-L5. There is deliberately NO smaller default:
# a new precision with larger staged items must be measured (RSS probe) and get its own entry here before use.
BF16_ACT=('bf16','bfp8w')
BUDGETS={'staging_budget_bytes':6979321856,'prepare_host_budget_bytes':7516192768}
# Trace region: R50 tuned graph has ~2x the R18 operations -> 256 MiB (adapter max).
TRACE_REGION={'r50':268435456,'r101':268435456}   # R101: more ops than R50 -> adapter max 256 MiB


def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def generate(assets:Path,executor:Path,variants=('r18',),precision=None):
 rows=[];cpus=sorted(os.sched_getaffinity(0));executor_hash=digest(executor)
 for variant in variants:
  # Per-precision R50 assets/configs (2026-11-16) live in <assets>/r50_<precision> with ids suffixed -<precision>;
  # R18 keeps its layout (<assets>/r18, plain ids). The old plain r50/r101 layout (FP32-conv / untuned, unqualified) was
  # removed 2026-11-18: --precision is required for r50/r101.
  if precision and variant not in ('r50','r101'):raise ValueError('--precision applies to r50/r101 only')
  if variant in ('r50','r101') and not precision:raise ValueError(f'{variant}: --precision bf16|bfp8w is required')
  tag=f'-{precision}' if precision else '';d=assets/(f'{variant}_{precision}' if precision else variant);plan=d/(variant+'.plan.json');weights=d/(variant+'.weights.f32');record=d/'compile_record.json'
  pdata=json.loads(plan.read_text())
  if pdata.get('variant')!=variant:raise ValueError(f'{plan}: variant identity mismatch')
  if precision and pdata.get('precision')!=precision:raise ValueError(f'{plan}: precision {pdata.get("precision")} != {precision}')
  if pdata.get('precision') not in BF16_ACT:raise ValueError(f'{plan}: precision {pdata.get("precision")} not supported (BF16 activations only)')
  declared=PRECISION_DECLARATION[pdata['precision']]
  budgets=dict(BUDGETS)
  wid=f'deeplabv3plus-{variant}-cityscapes-v1-eagle-n{tag}';did=f'eagle-n-{variant}{tag}'
  dut={'schema_version':1,'dut_config_id':did,'adapter':'bos_eagle_n','device':'0','max_inflight':1,
       'runtime_options':{'device_id':0,'executor':str(executor.resolve()),'executor_sha256':executor_hash,
        'lock_path':str((assets/'device0.lock').resolve()),'trace_region_bytes':TRACE_REGION.get(variant,67108864) if precision else 67108864,**budgets}}
  work=yaml.safe_load((ROOT/f'configs/workloads/deeplabv3plus-{variant}-cityscapes-v1.yaml').read_text())
  work['workload_id']=wid;work['precision']=declared
  work['options']['device_program']={'language':'bos_ttnn_native_plan','variant':variant,'plan':str(plan.resolve()),'plan_sha256':digest(plan),
    'weights':str(weights.resolve()),'weights_sha256':digest(weights),'input_bytes':25165824,'output_bytes':2097152,'task_output_bytes':2097152,
    'compile_record':str(record.resolve()),'compile_record_sha256':digest(record),'qualification':str((d/'native_qualification.json').resolve())}
  paths={ROOT/f'configs/duts/{did}.yaml':dut,ROOT/f'configs/workloads/{wid}.yaml':work}
  for stage in ('capture','baseline'):
   config=yaml.safe_load((ROOT/f'configs/experiments/deeplabv3plus-{variant}-{stage}.yaml').read_text())
   config['experiment_id']=f'deeplabv3plus-{variant}-eagle-n{tag}-{stage}';config['workload']=f'../workloads/{wid}.yaml';config['dut']=f'../duts/{did}.yaml';config['realtime']['cpu_affinity']=cpus
   if stage=='baseline':
    # Never copy GPU golden into the NPU baseline. Filled only after a real NPU capture.
    golden=d/'npu-capture.golden.jsonl'
    # SDV-BMT requires a relative output_reference.path: the golden used is the copy in configs/references/
    # (Boryoung: same name as the GPU golden with "eagle-n" instead of "gpu"); it must equal the captured golden.
    ref=ROOT/f'configs/references/cityscapes-val500.deeplabv3plus-{variant}-cityscapes-v1.eagle-n{tag}-capture.golden.jsonl'
    if golden.exists():
     if not ref.exists() or digest(ref)!=digest(golden):raise SystemExit(f'{ref} missing or differs from {golden}')
     config['evaluation']['output_reference'].update(path=f'../references/{ref.name}',sha256=digest(ref))
    else:config['evaluation']['output_reference'].update(path=str(golden.resolve()),sha256='REQUIRES_REAL_NPU_GOLDEN_AFTER_L4')
   suffix='.yaml.in' if stage=='baseline' and not golden.exists() else '.yaml'
   path=ROOT/f'configs/experiments/{config["experiment_id"]}{suffix}';paths[path]=config
  for path,data in paths.items():
   header='# Generated by tools/eagle_n/configure.py; local experimental native Plan C.\n# Actual NPU execution requires matching native qualification; not provided by Python standalone PASS.\n'
   if 'baseline' in path.name and not (d/'npu-capture.golden.jsonl').exists():header+='# TEMPLATE ONLY, not runnable YAML: supply actual NPU golden after L4 and rerun configure.py.\n'
   path.write_text(header+yaml.safe_dump(data,sort_keys=False))
   rows.append(str(path.relative_to(ROOT)))
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,default=Path('/home/bmt/caontd/eagle_n_integration_assets'));p.add_argument('--executor',type=Path,default=Path('/home/bmt/caontd/eagle_n_integration_assets/libeagle_n_ttnn.so'));p.add_argument('--variant',action='append',choices=('r18','r50','r101'),help='generate only this variant; repeat to select several (default: r18)');p.add_argument('--precision',choices=BF16_ACT,help='required for r50/r101: use <assets>/<variant>_<precision> and -<precision> ids');a=p.parse_args()
 print(json.dumps({'generated':generate(a.assets,a.executor,tuple(a.variant) if a.variant else ('r18',),a.precision)},indent=2))
if __name__=='__main__':main()
