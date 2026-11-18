"""Real common C++ loop/codec/bundle/evaluator, using an explicitly MOCK executor.
Never a BOS/NPU correctness or performance result. Production rejects the DLL.
"""
from __future__ import annotations
import base64,hashlib,json,subprocess,os
from pathlib import Path
import pytest,yaml
from sdv_bmt.cli import main as cli
from sdv_bmt.contracts import labelmap
from sdv_bmt.contracts.records import DUTDescriptor
from sdv_bmt.contracts.status import CompletionModel,ExecutionKind,RunClockDomain
from sdv_bmt.dut.eagle_n.adapter import EagleNDUT
from sdv_bmt.dut.native import native_binary,native_identity
from sdv_bmt.dut.native_source import native_source_tree_sha256
from tests._helpers import REPO_ROOT as ROOT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@pytest.fixture(scope='module')
def contract_binaries(tmp_path_factory):
    out=tmp_path_factory.mktemp('eagle_contract');lib=out/'lib_contract_mock.so';exe=ROOT/'native/build/sdv-bmt-native-eagle-contract-mock'
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared',str(ROOT/'tests/eagle_n/contract_mock.cpp'),'-o',str(lib)],check=True)
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',str(ROOT/'tests/eagle_n/contract_main.cpp'),'-o',str(exe),'-ldl'],check=True)
    Path(str(exe)+'.source-tree.sha256').write_text(native_source_tree_sha256(ROOT/'native')+'\n')
    actual=out/'actual_backend';subprocess.run(['g++','-std=c++17','-O2',str(ROOT/'native/src/main_eagle_n.cpp'),'-o',str(actual),'-ldl'],check=True)
    return lib,exe,actual

def test_actual_backend_rejects_mock_executor_before_prepare(contract_binaries):
    lib,_,actual=contract_binaries;r=subprocess.run([str(actual),'describe',json.dumps({'executor':str(lib),'executor_sha256':sha(lib)})],capture_output=True,text=True)
    assert r.returncode!=0 and 'mismatch' in r.stderr+r.stdout

def test_dlopen_digest_mismatch_is_rejected(contract_binaries):
    lib,exe,_=contract_binaries;r=subprocess.run([str(exe),'describe',json.dumps({'executor':str(lib),'executor_sha256':'0'*64})],capture_output=True,text=True)
    assert r.returncode!=0 and 'SHA-256' in r.stderr+r.stdout

class ContractMockDUT(EagleNDUT):
    ADAPTER='eagle_n_contract_mock';BINARY='sdv-bmt-native-eagle-contract-mock'
    def describe(self):
        return DUTDescriptor(adapter=self.ADAPTER,adapter_version='TEST_ONLY',implementation_kind=ExecutionKind.MOCK,
            device='NO DEVICE: contract mock',completion_model=CompletionModel.POLLING,clock_domain=RunClockDomain.MONOTONIC,
            submit_semantics='MOCK: fills preallocated label map by input slot',complete_semantics='MOCK: output final after submit',
            device_identity=native_identity(native_binary(self.BINARY)))

def _synthetic_r18_workload(p):
    """R-18 Eagle-N workload whose device_program points at small synthetic plan/weights/record files in p.
    The MOCK executor never reads the model, so the test needs no local Eagle-N assets (CI runners have none)."""
    work=yaml.safe_load((ROOT/'configs/workloads/deeplabv3plus-r18-cityscapes-v1-eagle-n.yaml').read_text());dp=work['options']['device_program']
    plan=p/'synthetic.plan.json';plan.write_text(json.dumps({'schema':'sdv-bmt-eagle-n-ttnn-plan/1','onnx_sha256':work['artifact_sha256'],'variant':'r18','precision':'bf16','ops':[],'test_only':True}))
    weights=p/'synthetic.weights.f32';weights.write_bytes(bytes(64))
    record=p/'synthetic.compile_record.json';record.write_text(json.dumps({'plan_sha256':sha(plan),'weights_sha256':sha(weights),'test_only':True}))
    dp.update(plan=str(plan),plan_sha256=sha(plan),weights=str(weights),weights_sha256=sha(weights),compile_record=str(record),
              compile_record_sha256=sha(record),qualification=str(p/'no_qualification.json'))
    wp=p/'workload.yaml';wp.write_text(yaml.safe_dump(work,sort_keys=False));return wp

def test_cli_to_native_bundle_and_miou_without_npu(contract_binaries,tmp_path,monkeypatch,capsys):
    lib,_,_=contract_binaries;monkeypatch.setitem(cli.DUT_ADAPTERS,ContractMockDUT.ADAPTER,ContractMockDUT)
    p=tmp_path;rows=[];W,H=2048,1024
    for i in range(2):
        payload=b'P6\n2048 1024\n255\n'+bytes([20+i,30+i,40+i])*(W*H);path=p/f'{i}.ppm';path.write_bytes(payload)
        annotation={'encoding':labelmap.CODEC_ID,'width':W,'height':H,'labels_b64':base64.b64encode(labelmap.encode(bytes([i])*(W*H),W,H)).decode('ascii')}
        rows.append({'source_item_id':f'contract-only-{i}','payload_file':path.name,'payload_sha256':sha(path),'annotation':annotation,'metadata':{'test_only':True}})
    idx=p/'index.jsonl';idx.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in rows))
    inp=yaml.safe_load((ROOT/'configs/inputs/cityscapes-val500.yaml').read_text());inp['input_id']='contract-mock-two-items';inp['source']={'index':idx.name,'index_sha256':sha(idx)};inp['selection']={'start':0,'count':2}
    ip=p/'input.yaml';ip.write_text(yaml.safe_dump(inp))
    dut=yaml.safe_load((ROOT/'configs/duts/eagle-n-r18.yaml').read_text());dut['adapter']=ContractMockDUT.ADAPTER;dut['runtime_options'].update(executor=str(lib),executor_sha256=sha(lib),lock_path=str(p/'lock'))
    dp=p/'dut.yaml';dp.write_text(yaml.safe_dump(dut))
    plan=yaml.safe_load((ROOT/'configs/experiments/deeplabv3plus-r18-eagle-n-capture.yaml').read_text());plan.update(experiment_id='eagle-native-contract-MOCK',execution_kind='mock',input=ip.name,dut=dp.name,workload=_synthetic_r18_workload(p).name)
    plan['stress']['spec']=os.path.relpath(ROOT/'configs/stress/fake-virtual-axis.yaml',p);plan['realtime']['precondition_requests']=0
    ep=p/'experiment.yaml';ep.write_text(yaml.safe_dump(plan))
    assert cli.main(['validate',str(ep)])==0;capsys.readouterr()
    assert cli.main(['run',str(ep),'--out',str(p/'bundles')])==0
    result=json.loads(capsys.readouterr().out);manifests=list((p/'bundles').rglob('manifest.json'));assert len(manifests)==1
    run=manifests[0].parent;assert 'mock' in manifests[0].read_text().lower()
    assert (run/'events.jsonl').exists() and (run/'outputs.bin').stat().st_size>0 and (run/'quality_reference.jsonl').exists()
    assert cli.main(['evaluate',str(run)])==0;capsys.readouterr()
    metrics=list((run/'evaluations').rglob('metrics.json'));assert len(metrics)==1 and 'cityscapes' in metrics[0].read_text().lower()
    review=Path('/home/bmt/caontd/eagle_n_integration_review')
    if review.is_dir():(review/'CONTRACT_MOCK_BUNDLE.json').write_text(json.dumps({'execution_kind':'mock','npu_used':False,'bundle':str(run),'metrics':str(metrics[0]),'cli_result':result},indent=2)+'\n')

def test_real_backend_guards_are_exercised_with_mock_abi(contract_binaries,tmp_path):
    lib,_,_=contract_binaries;exe=tmp_path/'backend_guards'
    subprocess.run(['g++','-std=c++17','-O2',str(ROOT/'tests/eagle_n/backend_guards.cpp'),'-o',str(exe),'-ldl'],check=True)
    work=yaml.safe_load((ROOT/'configs/workloads/deeplabv3plus-r18-cityscapes-v1-eagle-n.yaml').read_text());p=work['options']['device_program']
    program={'plan_path':p['plan'],'plan_sha256':p['plan_sha256'],'weights_path':p['weights'],'weights_sha256':p['weights_sha256'],'artifact_sha256':work['artifact_sha256'],'input_bytes':25165824,'task_output_bytes':2097152}
    opts={'device_id':0,'executor':str(lib),'executor_sha256':sha(lib),'lock_path':str(tmp_path/'lock'),'staging_budget_bytes':25165824}
    r=subprocess.run([str(exe),json.dumps({'backend':{'options':opts},'program':program})],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    report=json.loads(r.stdout);assert report=={'test_only':True,'npu_used':False,'checks':15}
