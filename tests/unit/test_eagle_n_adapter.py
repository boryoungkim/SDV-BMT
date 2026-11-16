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
@pytest.mark.parametrize('model',['r18','r50','r101'])
def test_plugin_preserves_canonical_workload(model):
 base=load_yaml_strict(ROOT/f'configs/workloads/deeplabv3plus-{model}-cityscapes-v1.yaml')
 new=load_yaml_strict(ROOT/f'configs/workloads/deeplabv3plus-{model}-cityscapes-v1-eagle-n.yaml')
 a=copy.deepcopy(base);b=copy.deepcopy(new)
 for d in (a,b):
  for k in ('workload_id','precision'):d.pop(k)
  d['options'].pop('device_program')
 assert a==b
 w=DeepLabCityscapesWorkload(parse_strict(WorkloadSpec,new))
 assert w.input_contract().contract_id=='cityscapes-rgb-norm-chw2048x1024-f32'
 p=parse_strict(EagleNProgram,new['options']['device_program']);assert p.task_output_bytes==2097152
@pytest.mark.parametrize('model',['r18','r50','r101'])
def test_native_plan_identity_and_bindings(model):
 work=load_yaml_strict(ROOT/f'configs/workloads/deeplabv3plus-{model}-cityscapes-v1-eagle-n.yaml')
 p=work['options']['device_program'];plan=json.loads(Path(p['plan']).read_text());blob=Path(p['weights']).read_bytes()
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
# Variants whose native qualification receipt was generated from real L1 evidence (evidence log §13, 2026-11-15).
QUALIFIED={'r18'}
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
@pytest.mark.parametrize('model',['r18','r50','r101'])
def test_no_gpu_golden_or_fake_qualification_in_baseline_templates(model):
 p=ROOT/f'configs/experiments/deeplabv3plus-{model}-eagle-n-baseline.yaml.in'
 assert p.exists();s=p.read_text();assert 'gpu-capture' not in s and 'REQUIRES_REAL_NPU_GOLDEN' in s
 receipt=Path(f'/home/bmt/caontd/eagle_n_integration_assets/{model}/native_qualification.json')
 if model not in QUALIFIED:
  assert not receipt.exists();return
 # A receipt may exist only if it is genuine: pinned to the current executor/plan/weights and to existing L1 evidence.
 work=load_yaml_strict(ROOT/f'configs/workloads/deeplabv3plus-{model}-cityscapes-v1-eagle-n.yaml')['options']['device_program']
 dut=load_yaml_strict(ROOT/f'configs/duts/eagle-n-{model}.yaml')['runtime_options']
 if not receipt.exists():pytest.skip('NOT RUN: Eagle-N assets not present on this host')
 r=json.loads(receipt.read_text())
 assert r['passed'] is True and r['variant']==model and Path(work['qualification'])==receipt
 assert r['executor_sha256']==dut['executor_sha256']==_sha(dut['executor'])
 assert r['plan_sha256']==work['plan_sha256']==_sha(work['plan'])
 assert r['weights_sha256']==work['weights_sha256']==_sha(work['weights'])
 assert r['evidence_sha256'] and all(_sha(f)==h for f,h in r['evidence_sha256'].items())
 # the generated baseline must use the NPU golden, never the GPU one
 base=load_yaml_strict(ROOT/f'configs/experiments/deeplabv3plus-{model}-eagle-n-baseline.yaml')
 ref=base['evaluation']['output_reference']['path'];assert 'eagle-n-capture' in ref and 'gpu-capture' not in ref
def test_h04_exporter_fails_before_parsing_unknown_artifact(tmp_path):
 spec=importlib.util.spec_from_file_location('eagle_export',ROOT/'tools/eagle_n/export_plan.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 p=tmp_path/'wrong.onnx';p.write_bytes(b'not-the-assigned-model')
 with pytest.raises(ValueError,match='H04'):m.convert(p,'r18','bf16')
def test_capture_is_asap_with_original_timeout_and_no_output_reference():
 for m in ('r18','r50','r101'):
  p=load_yaml_strict(ROOT/f'configs/experiments/deeplabv3plus-{m}-eagle-n-capture.yaml')
  ref=load_yaml_strict(ROOT/f'configs/experiments/deeplabv3plus-{m}-capture.yaml')
  assert p['replay']==ref['replay'] and p['run']==ref['run']
  assert p.get('evaluation',{}).get('output_reference') is None

def _launcher(monkeypatch):
 spec=importlib.util.spec_from_file_location('eagle_container',ROOT/'tools/eagle_n/in_container.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 def forbidden(*a,**kw):raise AssertionError('No device/subprocess before qualification')
 monkeypatch.setattr(m.subprocess,'run',forbidden)
 return m
@pytest.mark.parametrize('model',sorted({'r18','r50','r101'}-QUALIFIED))
def test_actual_launcher_refuses_unqualified_graph_before_any_subprocess(monkeypatch,model):
 m=_launcher(monkeypatch)
 with pytest.raises(ValueError,match='qualification NOT RUN'):m.precheck(ROOT/f'configs/experiments/deeplabv3plus-{model}-eagle-n-capture.yaml')
@pytest.mark.parametrize('model',sorted(QUALIFIED))
def test_actual_launcher_refuses_stale_qualification_before_any_subprocess(monkeypatch,model):
 if not Path(f'/home/bmt/caontd/eagle_n_integration_assets/{model}/native_qualification.json').exists():pytest.skip('NOT RUN: Eagle-N assets not present on this host')
 m=_launcher(monkeypatch);monkeypatch.setattr(m,'sha',lambda path:'0'*64)  # executor/plan/weights no longer match the receipt
 with pytest.raises(ValueError,match='qualification stale or failed'):m.precheck(ROOT/f'configs/experiments/deeplabv3plus-{model}-eagle-n-capture.yaml')

def test_sdk_dependency_drift_blocks_before_native_describe(config,monkeypatch):
 import sdv_bmt.dut.eagle_n.adapter as a
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
 for variant in ('r50','r101'):
  work=load_yaml_strict(ROOT/f'configs/workloads/deeplabv3plus-{variant}-cityscapes-v1-eagle-n.yaml')
  with pytest.raises(ValueError,match='restricted to R18'):dut.backend_spec(parse_strict(WorkloadSpec,work))

@pytest.mark.parametrize('value',[1,'yes',None])
def test_unqualified_trial_requires_boolean(config,value):
 config['runtime_options']['trial_unqualified']=value
 with pytest.raises(ValueError):EagleNDUT(parse_strict(DUTConfig,config))
