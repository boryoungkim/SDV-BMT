"""Eagle-N interface tests; no NPU or vendor SDK required."""
from __future__ import annotations
import copy,hashlib,importlib.util,json,subprocess
from pathlib import Path
import pytest
from sdv_bmt.contracts.specs import DUTConfig,WorkloadSpec,parse_strict
from sdv_bmt.contracts.status import RunClockDomain,CompletionModel
from sdv_bmt.dut.eagle_n.adapter import EagleNDUT,EagleNOptions,EagleNProgram
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.deeplab import DeepLabCityscapesWorkload
from tests._helpers import REPO_ROOT
ROOT=REPO_ROOT
@pytest.fixture
def config():return load_yaml_strict(ROOT/'configs/duts/eagle-n-r18.yaml')
@pytest.mark.parametrize('key,value',[('device_id',1),('device_id',True),('trace_region_bytes',0),('executor','relative/executor.so'),('executor_sha256','f'*63),('staging_budget_bytes',1)])
def test_options_reject_invalid(config,key,value):
 data=copy.deepcopy(config['runtime_options']);data[key]=value
 with pytest.raises(ValueError):parse_strict(EagleNOptions,data)
def test_unknown_option_is_error(config):
 config['runtime_options']['invented_override']=True
 with pytest.raises(ValueError):EagleNDUT(parse_strict(DUTConfig,config))
@pytest.mark.parametrize('name',['r18','r50-bf16','r50-bfp8w','r101-bf16'])
def test_plugin_preserves_canonical_workload(name):
 model=CONFIGS[name][0]
 base=load_yaml_strict(ROOT/f'configs/workloads/deeplabv3plus-{model}-cityscapes-v1.yaml')
 new=load_yaml_strict(_paths(name)['workload'])
 a=copy.deepcopy(base);b=copy.deepcopy(new)
 for d in (a,b):
  for k in ('workload_id','precision'):d.pop(k)
  d['options'].pop('device_program')
 assert a==b
 w=DeepLabCityscapesWorkload(parse_strict(WorkloadSpec,new))
 assert w.input_contract().contract_id=='cityscapes-rgb-norm-chw2048x1024-f32'
 p=parse_strict(EagleNProgram,new['options']['device_program']);assert p.task_output_bytes==2097152
@pytest.mark.parametrize('name',['r18','r50-bf16','r50-bfp8w','r101-bf16'])
def test_native_plan_identity_and_bindings(name):
 model=CONFIGS[name][0]
 work=load_yaml_strict(_paths(name)['workload'])
 p=work['options']['device_program']
 if not Path(p['plan']).exists():pytest.skip('NOT RUN: Eagle-N assets not present on this host')
 plan=json.loads(Path(p['plan']).read_text());blob=Path(p['weights']).read_bytes()
 assert plan['onnx_sha256']==work['artifact_sha256'] and plan['logical_convs']=={'r18':37,'r50':70,'r101':121}[model]
 for op in plan['ops']:
  if op['kind']=='Conv':
   for key in ('weight','bias'):
    x=op[key];assert hashlib.sha256(blob[x['offset']:x['offset']+x['bytes']]).hexdigest()==x['sha256']
 assert plan['ops'][-1]['kind']=='Cast' and plan['ops'][-2]['kind']=='ArgMax'
def test_measurement_declares_sync_cast_and_no_fabricated_device_time(config):
 d=EagleNDUT(parse_strict(DUTConfig,config));m=d.measurement_declaration()
 assert m['device_timing'] is None and m['prepare_input_h2d']=='included'
 assert 'synchronous' in m['instrumentation']['submit_semantics']
 assert 'collect' in m['instrumentation']['output_conversion']
 assert d.capabilities().completion_model==CompletionModel.POLLING and not d.capabilities().device_timing
 assert len(d.realization_evidence(None))==8
# Eagle-N configurations: name -> (variant, config tag, assets dir). The plain r50/r101 configs (FP32-conv / untuned,
# never qualified) were removed 2026-11-18; only BF16-activation plans remain.
CONFIGS={'r18':('r18','','r18'),'r50-bf16':('r50','-bf16','r50_bf16'),'r50-bfp8w':('r50','-bfp8w','r50_bfp8w'),
         'r101-bf16':('r101','-bf16','r101_bf16')}
# Configurations whose native qualification receipt was generated from real L1 evidence (evidence log §13, §53, §69).
QUALIFIED={'r18','r50-bf16','r101-bf16'}
ASSETS=Path('/home/bmt/caontd/eagle_n_integration_assets')
def _paths(name):
 v,t,d=CONFIGS[name]
 return dict(capture=ROOT/f'configs/experiments/deeplabv3plus-{v}-eagle-n{t}-capture.yaml',
             template=ROOT/f'configs/experiments/deeplabv3plus-{v}-eagle-n{t}-baseline.yaml.in',
             baseline=ROOT/f'configs/experiments/deeplabv3plus-{v}-eagle-n{t}-baseline.yaml',
             workload=ROOT/f'configs/workloads/deeplabv3plus-{v}-cityscapes-v1-eagle-n{t}.yaml',
             dut=ROOT/f'configs/duts/eagle-n-{v}{t}.yaml',receipt=ASSETS/d/'native_qualification.json')
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def test_no_fp32_or_legacy_eagle_n_configs():
 for v in ('r50','r101'):
  assert not (ROOT/f'configs/duts/eagle-n-{v}.yaml').exists()
  assert not (ROOT/f'configs/workloads/deeplabv3plus-{v}-cityscapes-v1-eagle-n.yaml').exists()
  assert not (ROOT/f'configs/experiments/deeplabv3plus-{v}-eagle-n-capture.yaml').exists()
 for f in (ROOT/'configs').rglob('*eagle-n*'):assert 'fp32' not in f.read_text().lower().replace('fp32dst',''),f
@pytest.mark.parametrize('name',sorted(CONFIGS))
def test_no_gpu_golden_or_fake_qualification_in_baseline_templates(name):
 P=_paths(name)
 p=P['template']
 assert p.exists();s=p.read_text();assert 'gpu-capture' not in s and 'REQUIRES_REAL_NPU_GOLDEN' in s
 receipt=P['receipt']
 if name not in QUALIFIED:
  assert not receipt.exists();return
 # A receipt may exist only if it is genuine: pinned to the current executor/plan/weights and to existing L1 evidence.
 work=load_yaml_strict(P['workload'])['options']['device_program']
 dut=load_yaml_strict(P['dut'])['runtime_options']
 if not receipt.exists():pytest.skip('NOT RUN: Eagle-N assets not present on this host')
 r=json.loads(receipt.read_text())
 assert r['passed'] is True and r['variant']==CONFIGS[name][0] and Path(work['qualification'])==receipt
 assert r['executor_sha256']==dut['executor_sha256']==_sha(dut['executor'])
 assert r['plan_sha256']==work['plan_sha256']==_sha(work['plan'])
 assert r['weights_sha256']==work['weights_sha256']==_sha(work['weights'])
 assert r['evidence_sha256'] and all(_sha(f)==h for f,h in r['evidence_sha256'].items())
 # the generated baseline must use the NPU golden, never the GPU one
 base=load_yaml_strict(P['baseline'])
 ref=base['evaluation']['output_reference']['path'];assert f'eagle-n{CONFIGS[name][1]}-capture' in ref and 'gpu-capture' not in ref
def test_h04_exporter_fails_before_parsing_unknown_artifact(tmp_path):
 spec=importlib.util.spec_from_file_location('eagle_export',ROOT/'tools/eagle_n/export_plan.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 p=tmp_path/'wrong.onnx';p.write_bytes(b'not-the-assigned-model')
 with pytest.raises(ValueError,match='H04'):m.convert(p,'r18','bf16')
def test_capture_is_asap_with_original_timeout_and_no_output_reference():
 for name,(m,_,_) in CONFIGS.items():
  p=load_yaml_strict(_paths(name)['capture'])
  ref=load_yaml_strict(ROOT/f'configs/experiments/deeplabv3plus-{m}-capture.yaml')
  assert p['replay']==ref['replay'] and p['run']==ref['run']
  assert p.get('evaluation',{}).get('output_reference') is None

def _launcher(monkeypatch):
 spec=importlib.util.spec_from_file_location('eagle_container',ROOT/'tools/eagle_n/in_container.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 def forbidden(*a,**kw):raise AssertionError('No device/subprocess before qualification')
 monkeypatch.setattr(m.subprocess,'run',forbidden)
 return m
@pytest.mark.parametrize('name',sorted(set(CONFIGS)-QUALIFIED))
def test_actual_launcher_refuses_unqualified_graph_before_any_subprocess(monkeypatch,name):
 m=_launcher(monkeypatch)
 with pytest.raises(ValueError,match='qualification NOT RUN'):m.precheck(_paths(name)['capture'])
@pytest.mark.parametrize('name',sorted(QUALIFIED))
def test_actual_launcher_refuses_stale_qualification_before_any_subprocess(monkeypatch,name):
 if not _paths(name)['receipt'].exists():pytest.skip('NOT RUN: Eagle-N assets not present on this host')
 m=_launcher(monkeypatch);monkeypatch.setattr(m,'sha',lambda path:'0'*64)  # executor/plan/weights no longer match the receipt
 with pytest.raises(ValueError,match='qualification stale or failed'):m.precheck(_paths(name)['capture'])

def test_sdk_dependency_drift_blocks_before_native_describe(config,monkeypatch):
 import sdv_bmt.dut.eagle_n.adapter as a
 if not Path(config['runtime_options']['executor']).exists():pytest.skip('NOT RUN: Eagle-N executor not present on this host')
 dut=EagleNDUT(parse_strict(DUTConfig,config));real=a.file_sha
 def drift(path):
  return '0'*64 if path.endswith('/libtt_metal.so') else real(path)
 monkeypatch.setattr(a,'file_sha',drift)
 monkeypatch.setattr(a,'describe_native',lambda *args,**kwargs:pytest.fail('Must reject SDK drift before launching native process'))
 with pytest.raises(ValueError,match='SDK dependency drift'):dut.describe()


def test_unqualified_trial_is_explicit_and_declares_jit(config):
 config['runtime_options']['trial_unqualified']=True
 dut=EagleNDUT(parse_strict(DUTConfig,config))
 text=dut.measurement_declaration()['instrumentation']['submit_semantics']
 assert 'DIAGNOSTIC UNQUALIFIED' in text and 'JIT/allocation' in text
 for name in ('r50-bf16','r101-bf16'):
  work=load_yaml_strict(_paths(name)['workload'])
  with pytest.raises(ValueError,match='restricted to R18'):dut.backend_spec(parse_strict(WorkloadSpec,work))

@pytest.mark.parametrize('value',[1,'yes',None])
def test_unqualified_trial_requires_boolean(config,value):
 config['runtime_options']['trial_unqualified']=value
 with pytest.raises(ValueError):EagleNDUT(parse_strict(DUTConfig,config))

def test_exporter_and_adapter_agree_on_plan_precisions():
 import sdv_bmt.dut.eagle_n.adapter as a
 spec=importlib.util.spec_from_file_location('eagle_export2',ROOT/'tools/eagle_n/export_plan.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.PRECISIONS==a.PLAN_PRECISIONS and set(a.PRECISION_DECLARATION)=={p for v in a.PLAN_PRECISIONS.values() for p in v}
 assert a.PLAN_PRECISIONS['r18']==('bf16',)   # R18 stays BF16 (qualified plan)
 assert 'fp32_conv' not in a.PRECISION_DECLARATION   # FP32-conv plans removed 2026-11-18

def test_workload_precision_must_match_plan_precision(config):
 # The executor runs the precision written in the plan; a workload declaring anything else is refused.
 work=load_yaml_strict(_paths('r50-bf16')['workload'])
 if not Path(work['options']['device_program']['plan']).exists():pytest.skip('NOT RUN: Eagle-N assets not present on this host')
 plan=json.loads(Path(work['options']['device_program']['plan']).read_text())
 dut=EagleNDUT(parse_strict(DUTConfig,config))
 import sdv_bmt.dut.eagle_n.adapter as a
 assert work['precision']==a.PRECISION_DECLARATION[plan['precision']]
 dut.backend_spec(parse_strict(WorkloadSpec,work))
 work['precision']='bf16-conv-fp32dst' if plan['precision']!='bf16' else 'bfp8w-conv-bf16act-fp32dst'
 with pytest.raises(ValueError,match='precision declaration mismatch'):dut.backend_spec(parse_strict(WorkloadSpec,work))

def _exporter():
 spec=importlib.util.spec_from_file_location('eagle_export3',ROOT/'tools/eagle_n/export_plan.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ONNX={'r18':Path('/home/bmt/caontd/eagle_n_porting/deeplab_r18/models/deeplabv3plus_r18.onnx'),'r50':Path('/home/bmt/caontd/eagle_n_porting/deeplab_r50/models/deeplabv3plus_r50.onnx')}
R18_PLAN=Path('/home/bmt/caontd/eagle_n_integration_assets/r18/r18.plan.json')
def test_r50_tuned_launch_decisions_stay_inside_proven_envelope():
 # After the L1 OOM of l1_20261116T084925_329398Z: a non-R18 tuned plan must carry per-conv launch decisions, and the
 # R18 tuned config is used only where R18 already ran an equal-or-larger conv of the same category.
 if not all(p.exists() for p in (*ONNX.values(),R18_PLAN)):pytest.skip('NOT RUN: Eagle-N assets not present on this host')
 m=_exporter()
 r18,_=m.convert(ONNX['r18'],'r18','bf16');assert all('launch' not in o for o in r18['ops']) and 'launch_envelope' not in r18
 with pytest.raises(ValueError,match='--envelope'):m.convert(ONNX['r50'],'r50','bfp8w','tuned')
 with pytest.raises(ValueError,match='--envelope'):m.convert(ONNX['r18'],'r18','bf16',None,R18_PLAN)
 plan,_=m.convert(ONNX['r50'],'r50','bfp8w','tuned',R18_PLAN);env=m.envelope(json.loads(R18_PLAN.read_text()))
 convs=[o for o in plan['ops'] if o['kind']=='Conv'];assert all('launch' in o for o in convs)
 for o in convs:
  l=o['launch'];assert l['config'] in ('r18_tuned','standalone_default','native_default')
  cat,kind,_=m.r18_launch(o);me=m.launched(o,l['dw_slice'])
  inside=any(c==cat and k==kind and r['stride']==me['stride'] and all(r[x]>=me[x] for x in ('cin','cout','out_px','in_px','dil')) for c,k,r in env)
  if l['config']=='r18_tuned':assert inside,o['output']   # safety property: tuned config only inside the R18 envelope
  if l['config']=='standalone_default' and kind=='dw':assert l['dw_slice']==32
 d12=next(o for o in convs if '/aspp_modules.1/depthwise_conv/' in o['output'])   # the conv that hit the L1 OOM
 assert 0<d12['launch']['dw_slice']<=256
 assert plan['launch_envelope']['r18_tuned_convs']+plan['launch_envelope']['standalone_default_convs']==70

def test_proven_survey_dominance_rules(tmp_path):
 spec=importlib.util.spec_from_file_location('eagle_survey',ROOT/'tools/eagle_n/proven_survey.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.shapes(['Tensor(shape=Shape([1, 1, 32768, 1024]))','Shape([2048, 1024, 1, 1])'])==[(1,1,32768,1024),(2048,1024,1,1)]
 assert m.dominated(((1,1,512,1024),),((1,1,32768,1024),)) and not m.dominated(((1,1,32768,2048),),((1,1,32768,1024),))
 assert m.dominated(((512,1024),),((1,1,512,1024),))   # right-aligned rank padding
 def pf(path,ops):path.write_text(json.dumps({'ops':ops}));return str(path)
 call=lambda n,*s:{'n':n,'a':['Shape(['+', '.join(map(str,x))+'])' for x in s]}
 proven=pf(tmp_path/'p.json',[{'index':0,'kind':'Conv','output':'a','calls':[call('Matmul',(1,1,32768,1024),(1,1,1024,1024))]},
                              {'index':1,'kind':'Conv','output':'b','calls':[call('Matmul',(1,1,32768,1024),(1,1,1024,2048))]}])
 cand=pf(tmp_path/'c.json',[{'index':0,'kind':'Conv','output':'x','calls':[call('Matmul',(1,1,32768,1024),(1,1,1024,2048))]}])
 import sys as _s
 for last,expect in ((None,0),(0,1)):   # op #1 is proven only if the proven run got past it
  _s.argv=['s','--candidate',cand,'--proven',proven+(f':{last}' if last is not None else ''),'--out',str(tmp_path/'o.json')]
  assert m.main()==expect

def test_r50_bf16_decomposition_is_proven_and_equivalent():
 # layer4 + head are rewritten into blocks of proven shapes; verify() is the symbolic equivalence proof (raises on mismatch)
 if not all(p.exists() for p in (ONNX['r50'],R18_PLAN)):pytest.skip('NOT RUN: Eagle-N assets not present on this host')
 m=_exporter()
 plan,_=m.convert(ONNX['r50'],'r50','bf16','tuned',R18_PLAN)
 convs=[o for o in plan['ops'] if o['kind']=='Conv']
 assert plan['logical_convs']==70 and plan['decomposition']['executed_conv_ops']==len(convs)
 assert all(o['launch']['config'] in ('r18_tuned','native_default') for o in convs)        # no unproven standalone path left
 assert not any(o['kind']=='Conv' and o['shape'][1]>1024 for o in plan['ops'])            # no 2048-ch conv output remains
 recon,_=m.convert(ONNX['r50'],'r50','bf16','tuned',R18_PLAN,True)
 strip=lambda o:{**o,'launch':{k:v for k,v in o['launch'].items() if k!='reason'}} if 'launch' in o else o
 assert [strip(o) for o in recon['ops'][:101]]==[strip(o) for o in plan['ops'][:101]]       # ops #0-#100 = device-proven run

# Watchdog (tools/eagle_n/README.md): the Eagle-N capture plans keep replay asap and differ from the GPU-copy capture plan
# only in experiment_id, run.request_timeout_ns and run.drain_ns (config only; src/sdv_bmt/dut/native.py unchanged).
@pytest.mark.parametrize('name',['r18','r50-bf16','r101-bf16'])
def test_asap_t60_capture_plans_change_only_the_timeouts(name):
 base=load_yaml_strict(_paths(name)['capture']);t60=load_yaml_strict(Path(str(_paths(name)['capture']).replace('-capture.yaml','-capture-asap-t60.yaml')))
 assert t60['replay']['mode']=='asap' and t60['run']['request_timeout_ns']==t60['run']['drain_ns']==60_000_000_000
 assert t60['experiment_id']==base['experiment_id']+'-asap-t60'
 for d in (base,t60):d.pop('experiment_id');d['run'].pop('request_timeout_ns');d['run'].pop('drain_ns')
 assert base==t60
 # measure watchdog of run_window() for asap (n = 1): lead + T + drain + precondition x T + 120 s
 T=60.0;limit=t60['replay'].get('lead_ns',0)/1e9+T+T+t60['realtime']['precondition_requests']*T+120.0
 assert limit/(t60['realtime']['precondition_requests']+500)>4.0   # seconds available per request for 500 items

def test_run_script_refuses_watchdog_unsafe_capture_plans():
 spec=importlib.util.spec_from_file_location('eagle_checks',ROOT/'tools/eagle_n/eagle_n_checks.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.MIN_ASAP_TIMEOUT_S>1.0   # the 1 s GPU-copy timeout is refused for an asap capture on Eagle-N
 s=(ROOT/'tools/eagle_n/run_eagle_n.sh').read_text()
 for model in ('r18','r50','r101'):assert f'  {model})' in s
 assert s.count('-capture-asap-t60.yaml')==3 and '-eagle-n-capture.yaml' not in s and '-bf16-capture.yaml' not in s
