"""BOS Eagle-N native adapter. Manual TTNN model plan; synchronous semantics declared.

No Python model execution occurs in the native measured process. A native model
qualification receipt matching the model plan AND executor binary is mandatory for
benchmark runs. Explicit unqualified R18 trials run the graph directly, without trace.
Existing standalone Python results do not satisfy that native gate.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from ...contracts.records import DUTCapabilities, DUTDescriptor, RealizationEvidence
from ...contracts.specs import DUTConfig, WorkloadSpec, parse_strict
from ...contracts.status import CompletionModel, EvidenceValidity, ExecutionKind, RealizedCondition, RunClockDomain
from ..native import REPO_ROOT, NativeRealtimeAdapter, describe_native, native_binary, native_identity
H04 = {'r18':'d7776aa683a18ca4e09305360f7d7f2857bac1a665dbbbc3aee2c60348450969',
       'r50':'6177c7ee37f5ce9779b39cea5defee2f1262fca37e3154ad28b74eadfafdf476',
       'r101':'86d0f08c1852723892777c0216936e5b790d3b9345f0e83473fb21309c129745'}
def file_sha(path: str) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def private_path(path: str) -> str:
    p=Path(path)
    if not p.is_absolute() or p.resolve().is_relative_to(REPO_ROOT):
        raise ValueError('Eagle-N binary/assets paths must be absolute outside the source repository')
    return str(p.resolve())
@dataclass(frozen=True)
class EagleNOptions:
    executor: str
    executor_sha256: str
    lock_path: str
    trial_unqualified: bool = False
    device_id: int = 0
    trace_region_bytes: int = 67108864
    staging_budget_bytes: int = 4294967296
    prepare_host_budget_bytes: int = 4294967296
    def __post_init__(self):
        private_path(self.executor);private_path(self.lock_path)
        if type(self.trial_unqualified) is not bool:raise ValueError("trial_unqualified must be bool")
        if self.device_id!=0 or type(self.device_id) is not int:raise ValueError('Only qualified device_id 0')
        if len(self.executor_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.executor_sha256):raise ValueError('executor SHA-256 must be lowercase hex')
        if not 1048576<=self.trace_region_bytes<=268435456:raise ValueError('trace region outside conservative 1..256MiB range')
        if not 268435456<=self.prepare_host_budget_bytes<=8589934592:raise ValueError('prepare host budget outside 256MiB..8GiB range')
        if not 25165824<=self.staging_budget_bytes<=8589934592:raise ValueError('staging budget outside 24MiB..8GiB range')
@dataclass(frozen=True)
class EagleNProgram:
    language: str
    variant: str
    plan: str
    plan_sha256: str
    weights: str
    weights_sha256: str
    qualification: str
    input_bytes: int
    output_bytes: int
    task_output_bytes: int
    compile_record: str
    compile_record_sha256: str
    def __post_init__(self):
        if self.language!='bos_ttnn_native_plan' or self.variant not in H04:raise ValueError('Unsupported Eagle-N program')
        if (self.input_bytes,self.output_bytes,self.task_output_bytes)!=(25165824,2097152,2097152):raise ValueError('DeepLab raw byte contract mismatch')
        for p in (self.plan,self.weights,self.qualification,self.compile_record):private_path(p)
        for h in (self.plan_sha256,self.weights_sha256,self.compile_record_sha256):
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('invalid program digest')
class EagleNDUT(NativeRealtimeAdapter):
    ADAPTER='bos_eagle_n';VERSION='1-native-ttnn-plan';BINARY='sdv-bmt-native-eagle-n'
    PREPARE_TIMEOUT_S=1800.0
    def __init__(self,config: DUTConfig):
        super().__init__(config)
        self._options=parse_strict(EagleNOptions,config.runtime_options,'dut.runtime_options')
        if config.device!='0' or config.max_inflight!=1:raise ValueError('Eagle-N requires device "0", N=1')
        self._identity=None
    def describe(self) -> DUTDescriptor:
        o=self._options
        if file_sha(o.executor)!=o.executor_sha256:raise ValueError('executor digest mismatch')
        sdk_path=Path(o.executor+'.sdk.json')
        sdk=json.loads(sdk_path.read_text())
        for path,h in sdk['libraries_sha256'].items():
            if file_sha(path)!=h:raise ValueError('SDK dependency drift: '+path)
        binary=native_binary(self.BINARY)
        if self._identity is None:
            self._identity={**describe_native(binary,{'executor':o.executor,'executor_sha256':o.executor_sha256}),**native_identity(binary),'eagle_n.sdk_commit':sdk['sdk_commit'],'eagle_n.sdk_libraries':sdk['libraries_sha256'],'eagle_n.sdk_identity_sha256':file_sha(str(sdk_path))}
        if o.trial_unqualified:
            self._identity={**self._identity,'eagle_n.submit_semantics':'DIAGNOSTIC: direct graph and logical D2H; includes first-use JIT/allocation','eagle_n.trial_unqualified':True}
        if self._identity.get('mock'):raise ValueError('ACTUAL Eagle-N adapter rejects mock executor')
        semantics=('DIAGNOSTIC UNQUALIFIED: direct graph, first-use JIT/allocation included; H2D + graph + D2H' if o.trial_unqualified else 'synchronous native C++: staged-host H2D, captured TTNN graph replay, blocking final D2H; returns with output final')
        return DUTDescriptor(adapter=self.ADAPTER,adapter_version=self.VERSION,implementation_kind=ExecutionKind.ACTUAL,
            device='BOS Eagle-N device 0 (revision not independently observed)',completion_model=CompletionModel.POLLING,
            clock_domain=RunClockDomain.MONOTONIC,device_identity=self._identity,
            submit_semantics=semantics,
            complete_semantics='poll returns 1 after successful synchronous submit; host arrivals/timeouts are serviced only after submit returns')
    def capabilities(self):return DUTCapabilities(completion_model=CompletionModel.POLLING,max_inflight=1,virtual_timing=False,device_timing=False)
    def measurement_declaration(self):
        return {'prepare_input_h2d':'included','device_timing':None,'instrumentation':{'adapter':self.ADAPTER,
            'measurement_plane':'native:'+self.BINARY,'profiler':'none','device_timing':None,
            'input_conversion':'stage, canonical CHW FP32 to runtime NHWC/dtype; outside MEASURE',
            'submit_semantics':('DIAGNOSTIC UNQUALIFIED: H2D + direct graph including first-use JIT/allocation + D2H' if self._options.trial_unqualified else 'synchronous: H2D + graph replay + D2H'),'output_conversion':'collect: checked uint32 indices to uint8, preallocated buffer',
            'model_placement':'BOS TTNN NPU, no Python/model CPU fallback','buffer_policy':('DIAGNOSTIC: graph/host unpack allocate during submit; not steady-state qualified' if self._options.trial_unqualified else 'prepare/capture persistent; native allocation/resource qualification required')}}
    def backend_spec(self,workload:WorkloadSpec):
        if workload.implementation!='deeplabv3plus_cityscapes':raise ValueError('Eagle-N adapter supports DeepLab family only')
        p=parse_strict(EagleNProgram,workload.options.get('device_program',{}),'workload.options.device_program')
        if workload.artifact_sha256!=H04[p.variant]:raise ValueError('H04 digest differs from selected variant')
        if self._options.trial_unqualified and p.variant!='r18':raise ValueError('Unqualified direct trial is restricted to R18')
        # FP32-conv R-50 plans are refused since 2026-11-18 (never qualified); BF16 activations only.
        expected_precision={'r18':'bf16-conv-fp32dst','r101':'bf16-conv-fp32dst'}.get(p.variant)
        if expected_precision is None:raise ValueError('No supported precision for '+p.variant+' (FP32-conv removed)')
        if workload.precision!=expected_precision:raise ValueError('Executed precision declaration mismatch')
        for path,h in [(p.plan,p.plan_sha256),(p.weights,p.weights_sha256),(p.compile_record,p.compile_record_sha256)]:
            if file_sha(path)!=h:raise ValueError('Eagle-N program asset digest mismatch: '+path)
        plan=json.loads(Path(p.plan).read_text());record=json.loads(Path(p.compile_record).read_text())
        if plan.get('schema')!='sdv-bmt-eagle-n-ttnn-plan/1' or plan.get('onnx_sha256')!=workload.artifact_sha256:raise ValueError('native plan identity mismatch')
        if record.get('plan_sha256')!=p.plan_sha256 or record.get('weights_sha256')!=p.weights_sha256:raise ValueError('compile record mismatch')
        return {'program':{'kind':'bos_ttnn_native_plan','plan_path':p.plan,'plan_sha256':p.plan_sha256,'weights_path':p.weights,'weights_sha256':p.weights_sha256,
            'artifact_sha256':workload.artifact_sha256,'qualification_path':p.qualification,'input_bytes':p.input_bytes,'task_output_bytes':p.task_output_bytes}}
    def realization_evidence(self,window:Any):
        # No device counter is invented. Host copy/residency code is a declaration,
        # not independent observation of clocks, temperatures or SRAM occupancy.
        return tuple(RealizationEvidence(c,EvidenceValidity.NOT_OBSERVABLE,'dut:'+self.ADAPTER,
            'No independently validated BOS telemetry for this condition; see measurement declaration for copy policy',None,'device',None)
            for c in RealizedCondition if c.value in ('input_residency','buffer_state','data_movement','runtime_queue_state','device_frequency','thermal_state','competing_device_users','memory_pressure_bandwidth'))
