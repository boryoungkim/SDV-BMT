"""Explicitly opt-in BOS C++ tests. No SDK/device test is claimed on a portable host."""
from __future__ import annotations
import json,os,subprocess
from pathlib import Path
import pytest
from tests._helpers import REPO_ROOT as ROOT

@pytest.mark.skipif(os.environ.get('SDV_BMT_EAGLE_N_SDK_TESTS')!='1',reason='NOT RUN: opt in inside the pinned BOS SDK container')
def test_eagle_n_native_describe_loads_real_executor():
    import yaml
    opt=yaml.safe_load((ROOT/'configs/duts/eagle-n-r18.yaml').read_text())['runtime_options']
    r=subprocess.run([str(ROOT/'native/build/sdv-bmt-native-eagle-n'),'describe',json.dumps({'executor':opt['executor'],'executor_sha256':opt['executor_sha256']})],capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stderr
    data=json.loads(r.stdout.strip().splitlines()[-1]);assert not data['mock'] and data['accelerator_family']=='NPU' and data['eagle_n.device_id']==0

@pytest.mark.skipif(os.environ.get('SDV_BMT_EAGLE_N_DEVICE_TESTS')!='1',reason='NOT RUN: new C++ modules/stages/full model not yet qualified on device; no automatic NPU access')
@pytest.mark.parametrize('workload',['deeplabv3plus-r18-cityscapes-v1-eagle-n','deeplabv3plus-r50-cityscapes-v1-eagle-n-bf16','deeplabv3plus-r101-cityscapes-v1-eagle-n-bf16'])
def test_native_model_qualification_matches_current_artifacts(workload):
    import hashlib,yaml
    work=yaml.safe_load((ROOT/f'configs/workloads/{workload}.yaml').read_text())
    p=work['options']['device_program'];q=Path(p['qualification'])
    assert q.is_file(),'Complete native module/stage/full gates before enabling this test; standalone Python is not a native receipt'
    d=json.loads(q.read_text());assert d['passed']
    for name,field in [('plan','plan_sha256'),('weights','weights_sha256')]:assert d[field]==hashlib.sha256(Path(p[name]).read_bytes()).hexdigest()
