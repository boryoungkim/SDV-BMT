#!/usr/bin/env python3
"""Offline ONNX-to-BOS TTNN execution plan; preserves original H04 artifact.

Uses the existing SDV-BMT ONNX reader; never imports torch/ttnn or opens a device.
Only the static DeepLab contract is accepted. CPU evaluation is restricted to
shape expressions; all data operators remain in the emitted NPU graph.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/stage2b_g_real'))
import onnx_reader as R
H04 = {'r18': 'd7776aa683a18ca4e09305360f7d7f2857bac1a665dbbbc3aee2c60348450969',
       'r50': '6177c7ee37f5ce9779b39cea5defee2f1262fca37e3154ad28b74eadfafdf476',
       'r101': '86d0f08c1852723892777c0216936e5b790d3b9345f0e83473fb21309c129745'}
SCHEMA = 'sdv-bmt-eagle-n-ttnn-plan/1'
def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
def convert(path: Path, variant: str, precision: str):
    content = path.read_bytes()
    if sha(content) != H04[variant]: raise ValueError('H04 SHA-256 mismatch; no conversion')
    # fp32_conv (R-50 FP32 convs on the NPU) removed 2026-11-18: never qualified; BF16 only.
    if precision != {'r18':'bf16', 'r101':'bf16'}.get(variant): raise ValueError('Unsupported variant/precision combination')
    model = R.load(str(path))
    if model.opsets.get('', 0) != 17: raise ValueError('opset must be 17')
    inputs = [x for x in model.inputs if x.name not in model.initializers]
    if len(inputs) != 1 or tuple(inputs[0].shape) != (1,3,1024,2048) or inputs[0].elem_type != R.FLOAT: raise ValueError('canonical input mismatch')
    if len(model.outputs) != 1 or tuple(model.outputs[0].shape) != (1,1,1024,2048) or model.outputs[0].elem_type != R.UINT8: raise ValueError('raw output mismatch')
    shapes = {inputs[0].name: [1,3,1024,2048]}; constants = {}; ops = []; weights = bytearray(); weight_info = {}
    for name,t in model.initializers.items():
        if t.data_type == R.INT64: constants[name] = t.values()
    def put(name):
        if name in weight_info: return weight_info[name]
        t = model.initializers[name]
        if t.data_type != R.FLOAT: raise ValueError('weights must be folded FP32 constants')
        weights.extend(b'\0' * (-len(weights) % 256)); offset=len(weights);weights.extend(t.raw)
        weight_info[name]={'offset':offset,'bytes':len(t.raw),'shape':list(t.dims),'sha256':sha(t.raw)}
        return weight_info[name]
    for node in model.nodes:
        kind=node.op_type; names=list(node.inputs); out=node.outputs[0]
        if len(node.outputs)!=1: raise ValueError('one output per node required')
        if kind=='Constant':
            t=node.attrs.get('value')
            if not isinstance(t,R.Tensor) or t.data_type!=R.INT64: raise ValueError('only static INT64 shape constants accepted')
            constants[out]=t.values();continue
        if kind=='Shape': constants[out]=list(shapes[names[0]]);continue
        if kind=='Slice':
            if not all(n in constants for n in names):raise ValueError('data Slice not supported by exporter')
            a=list(constants[names[0]]);start=constants[names[1]];end=constants[names[2]];axes=constants[names[3]] if len(names)>3 else [0]
            steps=constants[names[4]] if len(names)>4 else [1]
            if len(start)!=1 or axes!=[0] or steps!=[1]:raise ValueError('unsupported shape Slice')
            constants[out]=a[start[0]:end[0]];continue
        if kind=='Concat' and node.attrs['axis']==0 and all(n in constants for n in names):
            constants[out]=sum((list(constants[n]) for n in names),[]);continue
        item={'kind':kind,'inputs':names,'output':out}
        s=shapes[names[0]];a=node.attrs
        if kind=='Conv':
            w=put(names[1]);b=put(names[2]);cout,cg,kh,kw=w['shape'];group=a.get('group',1)
            stride=a.get('strides',[1,1]);dil=a.get('dilations',[1,1]);pad=a.get('pads',[0,0,0,0])
            if s[1]!=cg*group or b['shape']!=[cout] or pad[:2]!=pad[2:] or a.get('auto_pad',b'NOTSET')!=b'NOTSET':raise ValueError('Conv geometry mismatch')
            oh=(s[2]+pad[0]+pad[2]-dil[0]*(kh-1)-1)//stride[0]+1;ow=(s[3]+pad[1]+pad[3]-dil[1]*(kw-1)-1)//stride[1]+1
            item.update(weight=w,bias=b,kernel=[kh,kw],stride=stride,dilation=dil,padding=pad,groups=group)
            shape=[1,cout,oh,ow];item['inputs']=names[:1]
        elif kind in ('Relu','Add'):
            if kind=='Add' and shapes[names[1]]!=s:raise ValueError('broadcast residual prohibited')
            shape=list(s)
        elif kind=='MaxPool':
            k=a.get('kernel_shape');st=a.get('strides',[1,1]);p=a.get('pads',[0,0,0,0])
            if k!=[3,3] or st!=[2,2] or p!=[1,1,1,1] or a.get('ceil_mode',0)!=0 or a.get('dilations',[1,1])!=[1,1]:raise ValueError('unsupported MaxPool')
            shape=[1,s[1],(s[2]+1)//2,(s[3]+1)//2]
        elif kind=='GlobalAveragePool':shape=[1,s[1],1,1]
        elif kind=='Concat':
            if a['axis']!=1:raise ValueError('data concat must be channels')
            if any(shapes[n][0]!=1 or shapes[n][2:]!=s[2:] for n in names):raise ValueError('concat geometry')
            shape=[1,sum(shapes[n][1] for n in names),*s[2:]]
        elif kind=='Resize':
            if len(names)!=4 or names[1] or names[2] or names[3] not in constants:raise ValueError('Resize static sizes required')
            if a.get('mode')!=b'linear' or a.get('coordinate_transformation_mode') not in (b'half_pixel',b'pytorch_half_pixel'):raise ValueError('Resize semantics differ')
            shape=list(constants[names[3]]);item['inputs']=names[:1]
            if shape[:2]!=s[:2] or shape[2]%s[2] or shape[3]%s[3]:raise ValueError('only exact integer upsample supported')
        elif kind=='ArgMax':
            if a.get('axis')!=1 or a.get('keepdims',1)!=1 or a.get('select_last_index',0)!=0:raise ValueError('ArgMax contract')
            shape=[1,1,s[2],s[3]]
        elif kind=='Cast':
            if a.get('to')!=R.UINT8 or names[0]!=ops[-1]['output'] or ops[-1]['kind']!='ArgMax' or out!=model.outputs[0].name:raise ValueError('only final uint8 storage conversion accepted')
            shape=list(s);item['placement']='host_collect_checked_uint32_to_uint8'
        else:raise ValueError('unsupported data operator '+kind)
        if min(shape)<=0:raise ValueError('invalid shape')
        item['shape']=shape;shapes[out]=shape;ops.append(item)
    count=sum(o['kind']=='Conv' for o in ops)
    if count!={'r18':37,'r50':70,'r101':121}[variant]:raise ValueError('logical Conv count mismatch')
    if ops[-1]['kind']!='Cast':raise ValueError('final Cast missing')
    plan={'schema':SCHEMA,'variant':variant,'precision':precision,'onnx_sha256':sha(content),
          'input':{'name':inputs[0].name,'shape':[1,3,1024,2048],'bytes':25165824},
          'output':{'name':model.outputs[0].name,'shape':[1,1,1024,2048],'bytes':2097152},
          'ops':ops,'logical_convs':count,'weights_sha256':sha(bytes(weights)),
          'model_placement':'BOS TTNN device, final storage cast in native collect',
          'execution_route':'manual native TTNN plan; not Forge artifact','qualification':'NOT_RUN'}
    return plan,bytes(weights)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--onnx',type=Path,required=True);p.add_argument('--variant',choices=H04,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--precision',choices=('bf16',))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.out.resolve().is_relative_to(ROOT):raise ValueError('Model assets must be outside the repository')
    expected={'r18':'bf16','r101':'bf16'}.get(a.variant)
    if expected is None:raise ValueError(f'{a.variant}: no supported precision in this version (R-50 FP32-conv removed 2026-11-18)')
    precision=a.precision or expected
    if precision!=expected:raise ValueError(f'{a.variant} requires exporter precision {expected}')
    plan,weights=convert(a.onnx,a.variant,precision)
    target=a.out/(a.variant+'.plan.json');target.write_text(canonical(plan)+'\n');(a.out/(a.variant+'.weights.f32')).write_bytes(weights)
    record={'schema':'eagle-n-offline-build-record/1','onnx':str(a.onnx.resolve()),'onnx_sha256':plan['onnx_sha256'],'plan_sha256':sha(target.read_bytes()),'weights_sha256':sha(weights),'tool_sha256':sha(Path(__file__).read_bytes()),'command':sys.argv,'device_executed':False,'status':'EXPORTED_NOT_QUALIFIED'}
    (a.out/'compile_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
