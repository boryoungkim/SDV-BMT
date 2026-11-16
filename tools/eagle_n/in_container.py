#!/usr/bin/env python3
"""Run the isolated Eagle-N workspace in the pinned BOS container.

No devices are exposed for build/describe/validate/evaluate. `l1 START:COUNT` runs the offline L1
numeric harness (tools/eagle_n/l1_native.py) with the device. `run` first requires
native qualification and a staging memory budget; it uses a private hugepage
mount backed by the existing pool. Never reserves RAM, resets or reloads drivers.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
EVID_FS='/home/bmt/caontd'  # evidence + private shm dir live on this filesystem
ROOT=Path(__file__).resolve().parents[2]
IMAGE='sha256:bb29c5af20279d1ad95805ebfb2061399d409778b3ba54d87b1f102235d6ec8d'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def precheck_device():
 if not Path('/dev/bos/0').exists():raise ValueError('device node absent')
 r=subprocess.run(['sudo','-n','fuser','/dev/bos/0'],capture_output=True,text=True)
 if r.returncode!=1:raise ValueError('NPU is busy or competing-user query failed: '+r.stdout+r.stderr)
def precheck(plan):
 sys.path.insert(0,str(ROOT/'src'))
 from sdv_bmt.runner.plan import load_plan
 rp=load_plan(plan);opt=rp.dut.runtime_options;p=rp.workload.options['device_program']
 if rp.dut.adapter!='bos_eagle_n':raise ValueError('only bos_eagle_n allowed by this launcher')
 trial=opt.get('trial_unqualified',False)
 if trial:
  if p['variant']!='r18' or rp.input_spec.selection.count!=1 or rp.plan.realtime.precondition_requests!=0 or 'trial' not in rp.plan.experiment_id:
   raise ValueError('Unqualified direct trial requires R18, one input, zero precondition requests and trial experiment id')
 else:
  q=Path(p['qualification'])
  if not q.is_file():raise ValueError('Native model qualification NOT RUN: use explicit single-image R18 trial or complete native functional gates')
  receipt=json.loads(q.read_text())
  if not receipt.get('passed') or any(receipt.get(k)!=v for k,v in [('executor_sha256',sha(opt['executor'])),('plan_sha256',sha(p['plan'])),('weights_sha256',sha(p['weights']))]):raise ValueError('native qualification stale or failed')
 count=rp.input_spec.selection.count
 if count is None:raise ValueError('bounded staging needs explicit input selection count')
 size=(12582912 if p['variant']=='r18' else 25165824)*count
 need=count*int(p['input_bytes'])+2*2**30;free=shutil.disk_usage(EVID_FS).free
 if free<need:raise ValueError(f'SDV-BMT staging file needs {need} bytes free on disk (inputs + 2 GiB), only {free}')
 if size>opt['staging_budget_bytes']:raise ValueError(f'staged host payload {size} exceeds budget {opt["staging_budget_bytes"]}; resolve residency/budget before NPU access')
 if not Path('/dev/bos/0').exists():raise ValueError('device node absent')
 r=subprocess.run(['sudo','-n','fuser','/dev/bos/0'],capture_output=True,text=True)
 if r.returncode!=1:raise ValueError('NPU is busy or competing-user query failed: '+r.stdout+r.stderr)
 return rp

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['build','build-runner','describe','validate','run','evaluate','l1']);p.add_argument('target',nargs='?');p.add_argument('--sdk',type=Path,default=Path('/home/bmt/tt-metal'));p.add_argument('--evidence',type=Path,default=Path('/home/bmt/caontd/eagle_n_integration_evidence'));p.add_argument('--kernel-cache',type=Path);p.add_argument('--wall-seconds',type=int,default=1800);p.add_argument('--rss-probe-slots',type=int,default=0,help='l1 only: extra staging slots for the RSS probe');a=p.parse_args()
 if a.rss_probe_slots and (a.mode!='l1' or not 0<a.rss_probe_slots<=64):p.error('--rss-probe-slots: l1 only, 1..64')
 if a.mode not in ('build','build-runner','describe') and not a.target:p.error('target plan/bundle is required')
 if a.kernel_cache is not None and not a.kernel_cache.resolve().is_relative_to(Path('/home/bmt/caontd')):raise ValueError('kernel-cache must stay inside caontd')
 if a.mode=='run':precheck(a.target)
 if a.mode=='l1':precheck_device()
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out=a.evidence/(a.mode+'_'+stamp);out.mkdir(parents=True,mode=0o700)
 view=out/'runtime/sdk_view';(view/'generated').mkdir(parents=True)
 for name in ('tt_metal','ttnn','runtime','tt_stl','tools'): (view/name).symlink_to(a.sdk/name,target_is_directory=True)
 (view/'build').symlink_to(a.sdk/'build_Release_distributed',target_is_directory=True)
 env={'HOME':str(out),'TMPDIR':str(out/'tmp'),'XDG_CACHE_HOME':str(out/'cache'),'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1','TT_METAL_HOME':str(view),'TT_METAL_CACHE':str(a.kernel_cache.resolve() if a.kernel_cache is not None else out/'metal_cache'),'ARCH_NAME':'blackhole','TT_METAL_DISABLE_L1_DATA_CACHE_RISCVS':'BR,NC,TR,ER','LD_LIBRARY_PATH':':'.join(str(a.sdk/'build_Release_distributed'/n) for n in ('ttnn','tt_metal','lib','tt_stl')),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
 for name in ('tmp','cache','metal_cache'):(out/name).mkdir()
 name='caontd-sdv-eagle-'+stamp.lower().replace('_','-')
 cmd=['sudo','-n','docker','run','--rm','--init','--name',name,'--read-only','--network','none','--user',f'{os.getuid()}:{os.getgid()}','--cap-drop','ALL','--ulimit','core=0:0',
 '--mount','type=bind,source=/home/bmt/caontd,target=/home/bmt/caontd','--mount',f'type=bind,source={a.sdk},target={a.sdk},readonly','--mount',f'type=bind,source={a.sdk},target=/tt-metal,readonly','--workdir',str(ROOT)]
 huge=out/'hugepages';mounted=False;code=1
 try:
  if a.mode in ('run','l1'):
   free=Path('/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages')
   if not free.exists() or int(free.read_text())<1:raise ValueError('existing 1GiB hugepage pool has no free page; no system reservation is changed')
   huge.mkdir(mode=0o700);subprocess.run(['sudo','-n','mount','-t','hugetlbfs','-o',f'pagesize=1G,uid={os.getuid()},gid={os.getgid()},mode=0700','hugetlbfs',str(huge)],check=True);mounted=True
   cmd+=['--cap-add','IPC_LOCK','--ulimit','memlock=-1:-1','--device','/dev/bos/0:/dev/bos/0:rwm','--mount',f'type=bind,source={huge},target=/dev/hugepages-1G']
   if a.mode=='run':
    # SDV-BMT (original native.py) writes the whole canonical-input staging file to /dev/shm when writable. The docker
    # default /dev/shm is 64 MB and a RAM tmpfs would need 11.7 GiB; bind a private disk directory instead (INTERNAL).
    shm=out/'shm';shm.mkdir(mode=0o700);cmd+=['--mount',f'type=bind,source={shm},target=/dev/shm']
  for k,v in env.items():cmd+=['--env',k+'='+v]
  if a.mode=='build':cmd+=['--entrypoint','/bin/bash',IMAGE,str(ROOT/'tools/eagle_n/build_native.sh')]
  elif a.mode=='build-runner':cmd+=['--entrypoint','/bin/bash',IMAGE,str(ROOT/'tools/eagle_n/build_runner.sh')]
  elif a.mode=='l1':cmd+=['--entrypoint',str(ROOT/'.venv/bin/python'),IMAGE,'-B',str(ROOT/'tools/eagle_n/l1_native.py'),'--range',a.target,'--out',str(out/'l1'),'--rss-probe-slots',str(a.rss_probe_slots)]
  elif a.mode=='describe':
   import yaml
   dut=Path(a.target) if a.target else ROOT/'configs/duts/eagle-n-r18.yaml';opts=yaml.safe_load(dut.read_text())['runtime_options']
   cmd+=['--entrypoint',str(ROOT/'native/build/sdv-bmt-native-eagle-n'),IMAGE,'describe',json.dumps({'executor':opts['executor'],'executor_sha256':opts['executor_sha256']})]
  else:
   args=[a.mode,str(Path(a.target).resolve())]
   if a.mode=='run':args+=['--out',str(out/'bundles')]
   cmd+=['--entrypoint',str(ROOT/'.venv/bin/python'),IMAGE,'-B','-m','sdv_bmt.cli.main',*args]
  (out/'command.json').write_text(json.dumps({'command':cmd,'device_exposed':a.mode in ('run','l1'),'reset':False},indent=2)+'\n')
  with (a.evidence/'COMMANDS.log').open('a') as log:log.write(json.dumps({'utc':stamp,'command':cmd,'folder':str(out)})+'\n')
  with (out/'run.log').open('w') as log:
   try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.wall_seconds).returncode
   except subprocess.TimeoutExpired:
    subprocess.run(['sudo','-n','docker','stop','--time','5',name],capture_output=True,timeout=20);code=124
  print((out/'run.log').read_text(errors='replace')[-6000:])
 finally:
  if mounted:
   probe=subprocess.run(['sudo','-n','docker','inspect','--format','{{.State.Running}}',name],capture_output=True,text=True)
   if probe.returncode==0 and probe.stdout.strip()=='true':raise RuntimeError('container still running; private hugepage mount deliberately preserved')
   for f in huge.iterdir():
    if f.is_symlink() or not f.is_file() or not(f.name=='tenstorrent' or f.name.startswith('device_0_channel_')):raise RuntimeError('unexpected private hugepage file; preserve mount')
    f.unlink()
   subprocess.run(['sudo','-n','umount',str(huge)],check=True)
  shm=out/'shm';left=[]
  if shm.is_dir():  # remove a leftover SDV-BMT staging file (11.7 GiB); keep small spec files as evidence
   for f in shm.glob('sdv-bmt-staging-*.bin'):left.append(f.name);f.unlink()
  (out/'status.json').write_text(json.dumps({'exit_code':code,'removed_leftover_staging':left,'device_exposed':a.mode in ('run','l1'),'npu_reset':False,'private_mount_removed':not mounted or not os.path.ismount(huge)},indent=2)+'\n')
 print('Evidence:',out);return code
if __name__=='__main__':
 try:raise SystemExit(main())
 except (ValueError,OSError) as e:print('REJECTED BEFORE DEVICE ACCESS:',str(e),file=sys.stderr);raise SystemExit(2)
