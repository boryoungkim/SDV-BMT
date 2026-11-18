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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import onnx_reader as R
H04 = {'r18': 'd7776aa683a18ca4e09305360f7d7f2857bac1a665dbbbc3aee2c60348450969',
       'r50': '6177c7ee37f5ce9779b39cea5defee2f1262fca37e3154ad28b74eadfafdf476',
       'r101': '86d0f08c1852723892777c0216936e5b790d3b9345f0e83473fb21309c129745'}
SCHEMA = 'sdv-bmt-eagle-n-ttnn-plan/1'
# Allowed plan precisions per variant (first = default). R50: bf16 / bfp8w (bfloat8_b conv weights) allowed by
# Boryoung 2026-11-16. The executor reads 'precision' and 'graph' from the plan. fp32_conv (FP32 convs on the NPU, the
# standalone V10.1 profile) was removed 2026-11-18: never qualified, 2x staging; BF16 activations only.
PRECISIONS = {'r18': ('bf16',), 'r50': ('bf16', 'bfp8w'), 'r101': ('bf16',)}
def sha(data): return hashlib.sha256(data).hexdigest()
# R50/R101 tuned graphs: per-conv launch decision written into the plan ('launch'), so the executor does not guess.
# A conv keeps the R18 tuned Conv2dConfig only if it is DOMINATED by a conv the R18 tuned graph already ran on the NPU
# (same config category/kind/stride, cin, cout, output and input pixels, dilation all <= that R18 conv). Every other
# conv uses the configuration the standalone R50 run (V10.1, 2026-11-12) completed on the NPU: default Conv2dConfig,
# depthwise split into 32-channel slices. Heuristic L1 envelope (INTERNAL rule, 2026-11-16) after the L1 OOM of
# l1_20261116T084925_329398Z; it is not an SDK guarantee. R18 plans carry no 'launch' key and are unchanged.
R18_TUNED_PLAN_SHA = '1f9f4cf648b2d6dfc6f7cc6daf71d2c7445fd3e290cb4cefac095fa0192fb614'
ENVELOPE_RULE = 'r18-tuned-dominance/1'
def r18_launch(op):
    """How the R18 tuned executor rules launch one conv: (category, kind, depthwise slice width or 0)."""
    cout,cin_g,kh,_=op['weight']['shape'];g=op['groups'];cin=cin_g*g;name=op['output'];dil=op['dilation'][0]
    depth=g>1 and g==cin==cout
    stem=any(s in name for s in ('/backbone/stem/stem.0/','/backbone/stem/stem.3/','/backbone/stem/stem.6/'))
    sep_dw='/sep_bottleneck.0/depthwise_conv/' in name or '/sep_bottleneck.1/depthwise_conv/' in name
    cat='height' if sep_dw else ('abh256_height' if stem else 'abh256')
    width=0
    if depth and cin>32 and dil>=24:width=128 if '/aspp_modules.2/depthwise_conv/' in name else 32
    return cat,('dw' if depth else ('pw' if kh==1 else 'k%d'%kh)),width
def launched(op,width):
    cout,cin_g,_,_=op['weight']['shape'];cin=cin_g*op['groups'];c=width or cin;st=op['stride'][0];px=op['shape'][2]*op['shape'][3]
    return {'cin':c,'cout':c if width else cout,'dil':op['dilation'][0],'stride':st,'out_px':px,'in_px':px*st*st}
def envelope(r18_plan):
    env=[]
    for op in r18_plan['ops']:
        if op['kind']!='Conv':continue
        cat,kind,width=r18_launch(op);env.append((cat,kind,launched(op,width)))
    return env
# Preflight calibration overrides (INTERNAL, evidence log §41): convs whose preflight cumulative L1 peak exceeded the largest
# value measured on the R18 tuned graph although they are inside the R18 shape envelope. stem.6 with bfloat8_b weights:
# 1,423,104 B/core (l1_20261116T123426_670518Z) > R18 max 1,412,928 B/core (l1_20261116T123418_025184Z) -> standalone default.
CALIBRATION_OVERRIDES = {'bfp8w': ('/backbone/stem/stem.6/',)}
# Convs that already COMPLETED on the device in exactly the native_default form (one default-config SDK call) in the bf16
# run l1_20261116T131610_455874Z, ops #0-#100 = stem..layer3 (evidence log §44/§45). Proven only for bf16 weights.
PROVEN_NATIVE = {'bf16': ('/backbone/stem/', '/backbone/layer1/', '/backbone/layer2/', '/backbone/layer3/')}
def launch_decision(op,env,precision=None):
    if any(k in op['output'] for k in CALIBRATION_OVERRIDES.get(precision,())):
        cout,cin_g,_,_=op['weight']['shape']
        return {'config':'standalone_default','dw_slice':0,'reason':'preflight calibration override'}
    d=_launch_decision(op,env)
    if d['config']=='standalone_default' and any(k in op['output'] for k in PROVEN_NATIVE.get(precision,())):
        return {'config':'native_default','dw_slice':d['dw_slice'],'reason':'completed on device in this form (bf16 run ops #0-#100)'}
    return d
def _launch_decision(op,env):
    cat,kind,width=r18_launch(op);cout,cin_g,_,_=op['weight']['shape'];cin=cin_g*op['groups']
    if kind=='dw' and not width and cin>512 and op['dilation'][0]>1:width=256   # wide dilated ASPP depthwise (R18 ran 512)
    me=launched(op,width)
    if any(c==cat and k==kind and r['stride']==me['stride'] and all(r[x]>=me[x] for x in ('cin','cout','out_px','in_px','dil')) for c,k,r in env):
        return {'config':'r18_tuned','dw_slice':width}
    return {'config':'standalone_default','dw_slice':32 if kind=='dw' and cin>32 else 0}
def native_proven_1x1(ops):
    """(M,K,N) of the bf16 native_default 1x1 stride-1 convs that completed on the device (ops #0-#100, §44)."""
    out=[]
    for o in ops:
        if o['kind']=='Conv' and o.get('launch',{}).get('config')=='native_default' and o['kernel']==[1,1] and o['stride']==[1,1] and o['groups']==1:
            cout,cin,_,_=o['weight']['shape'];out.append((o['shape'][2]*o['shape'][3],cin,cout))
    return out
def decompose_r50_bf16(ops,weights,env,input_name):
    from decompose_proven import Decomposer,Weights
    native=native_proven_1x1(ops)
    def decide(op,M,K,N,depthwise=False):
        kh,kw=op['kernel']
        syn=dict(op,groups=(K if depthwise else 1),shape=[1,N,op['shape'][2],op['shape'][3]],
                 weight=dict(op['weight'],shape=[N,1 if depthwise else K,kh,kw]))
        cat,kind,width=r18_launch(syn);me=launched(syn,width)
        if any(c==cat and k==kind and r['stride']==me['stride'] and all(r[x]>=me[x] for x in ('cin','cout','out_px','in_px','dil')) for c,k,r in env):
            return {'config':'r18_tuned','dw_slice':width}
        if not depthwise and kh==kw==1 and op['stride']==[1,1] and any(m>=M and k>=K and n>=N for m,k,n in native):
            return {'config':'native_default','dw_slice':0}
        return None
    from decompose_proven import verify
    new=Decomposer(ops,Weights(weights),decide,native).run({'name':input_name,'shape':[1,3,1024,2048]})
    report=verify(ops,new,weights,input_name)   # symbolic equivalence proof; raises on any mismatch
    print('DECOMPOSITION VERIFIED',report,file=sys.stderr)
    return new
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
def convert(path: Path, variant: str, precision: str, graph: str | None = None, envelope_plan: Path | None = None, reconstruct: bool = False):
    content = path.read_bytes()
    if sha(content) != H04[variant]: raise ValueError('H04 SHA-256 mismatch; no conversion')
    if precision not in PRECISIONS[variant]: raise ValueError('Unsupported variant/precision combination')
    if graph not in (None, 'base', 'tuned'): raise ValueError('Unsupported graph')
    per_op = variant != 'r18' and graph == 'tuned'
    if per_op != (envelope_plan is not None): raise ValueError('--envelope (the R18 tuned plan) is required exactly for non-R18 tuned graphs')
    if per_op:
        raw = envelope_plan.read_bytes()
        if sha(raw) != R18_TUNED_PLAN_SHA: raise ValueError('envelope plan is not the qualified R18 tuned plan')
        env = envelope(json.loads(raw))
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
            if per_op:
                item['launch']=launch_decision({**item,'shape':shape},env,None if reconstruct else precision)
                if reconstruct and item['launch']['config']=='standalone_default':item['launch']={'config':'native_default','dw_slice':item['launch']['dw_slice']}
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
    decomposed=False
    # R101 (2026-11-17): stem/layer1/layer2/layer4/head have exactly the R50 shapes, layer3 repeats the R50 layer3 block
    # (23 instead of 6 blocks) -> same per-conv launch rules and the same proven-block decomposition of layer4 + head.
    if per_op and variant in ('r50','r101') and precision=='bf16' and not reconstruct:
        ops=decompose_r50_bf16(ops,weights,env,inputs[0].name);decomposed=True
    if ops[-1]['kind']!='Cast':raise ValueError('final Cast missing')
    plan={'schema':SCHEMA,'variant':variant,'precision':precision,'onnx_sha256':sha(content),
          'input':{'name':inputs[0].name,'shape':[1,3,1024,2048],'bytes':25165824},
          'output':{'name':model.outputs[0].name,'shape':[1,1,1024,2048],'bytes':2097152},
          'ops':ops,'logical_convs':count,'weights_sha256':sha(bytes(weights)),
          'model_placement':'BOS TTNN device, final storage cast in native collect',
          'execution_route':'manual native TTNN plan; not Forge artifact','qualification':'NOT_RUN'}
    if graph is not None: plan['graph']=graph   # absent = executor default (r18 tuned, others base)
    if reconstruct: plan['reconstruction']='bf16 device run l1_20261116T131610_455874Z (ops #0-#100 proven, #101 hung); NEVER run'
    if decomposed: plan['decomposition']={'rule':'proven-block-decomposition/1','executed_conv_ops':sum(o['kind']=='Conv' for o in ops),
        'note':'layer4+head rewritten into blocks of R-18 / R-50-layer3 proven shapes (tools/eagle_n/decompose_proven.py)'}
    if per_op: plan['launch_envelope']={'rule':ENVELOPE_RULE,'r18_plan_sha256':R18_TUNED_PLAN_SHA,
        'r18_tuned_convs':sum(o.get('launch',{}).get('config')=='r18_tuned' for o in ops),
        'standalone_default_convs':sum(o.get('launch',{}).get('config')=='standalone_default' for o in ops),
        'native_default_convs':sum(o.get('launch',{}).get('config')=='native_default' for o in ops)}
    return plan,bytes(weights)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--onnx',type=Path,required=True);p.add_argument('--variant',choices=H04,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--precision',choices=('bf16','bfp8w'));p.add_argument('--graph',choices=('base','tuned'))
    p.add_argument('--envelope',type=Path,help='non-R18 tuned graphs: the qualified R18 tuned plan (per-conv launch decisions)')
    p.add_argument('--reconstruct-proven-run',action='store_true',help='survey only: the r50 bf16 plan exactly as run by l1_20261116T131610_455874Z')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.out.resolve().is_relative_to(ROOT):raise ValueError('Model assets must be outside the repository')
    precision=a.precision or PRECISIONS[a.variant][0]
    if precision not in PRECISIONS[a.variant]:raise ValueError(f'{a.variant} allows exporter precision {PRECISIONS[a.variant]}')
    plan,weights=convert(a.onnx,a.variant,precision,a.graph,a.envelope,a.reconstruct_proven_run)
    target=a.out/(a.variant+'.plan.json');target.write_text(canonical(plan)+'\n');(a.out/(a.variant+'.weights.f32')).write_bytes(weights)
    record={'schema':'eagle-n-offline-build-record/1','onnx':str(a.onnx.resolve()),'onnx_sha256':plan['onnx_sha256'],'plan_sha256':sha(target.read_bytes()),'weights_sha256':sha(weights),'tool_sha256':sha(Path(__file__).read_bytes()),'command':sys.argv,'device_executed':False,'status':'EXPORTED_NOT_QUALIFIED'}
    (a.out/'compile_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
