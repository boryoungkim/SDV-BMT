#!/usr/bin/env python3
"""Compare a real single-image bundle label map with immutable standalone goldens.

No NPU access; no NumPy dependency. Label agreement is not Cityscapes val500 mIoU.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, struct
from pathlib import Path
from sdv_bmt.contracts.labelmap import decode
from sdv_bmt.observation.reader import load_run_bundle, fold_requests, read_output_blob

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def npy_u8(p):
    data=p.read_bytes()
    if data[:6]!=b'\x93NUMPY': raise ValueError('not NPY')
    major=data[6];start=10 if major==1 else 12
    n=struct.unpack_from('<H' if major==1 else '<I',data,8)[0]
    h=ast.literal_eval(data[start:start+n].decode('latin1'))
    if h['descr'] not in ('|u1','<u1') or h['fortran_order']: raise ValueError('expected C-order uint8 golden')
    raw=data[start+n:]
    if len(raw)!=2097152: raise ValueError('golden shape differs')
    return raw

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    bundle=load_run_bundle(a.bundle);rows=fold_requests(bundle.events)
    done=[c for _,c in rows.values() if c is not None and c.output_ref is not None]
    if len(done)!=1: raise ValueError('expected exactly one final output')
    ref=done[0].output_ref;blob=read_output_blob(a.bundle,ref)
    if hashlib.sha256(blob).hexdigest()!=ref.sha256: raise ValueError("output blob checksum mismatch")
    item,w,h,raw=decode(blob)
    if (w,h,len(raw))!=(2048,1024,2097152) or max(raw)>=19: raise ValueError('invalid output')
    base=Path('/home/bmt/caontd/eagle_n_porting/deeplab_r18');sums=json.loads((base/'golden/SHA256SUMS.json').read_text())
    comparisons=[]
    for name,path in [('ORT FP32 golden',base/'golden/ort_labels_uint8.npy'),('PyTorch FP32 golden',base/'golden/labels_uint8.npy'),('previous standalone NPU BF16',base/'results/20261111T052211_877089Z/labels_npu_uint8.npy')]:
        if path.parent.name=='golden' and sha(path)!=sums[path.name]: raise ValueError('golden checksum mismatch')
        ref=npy_u8(path);equal=sum(x==y for x,y in zip(raw,ref));comparisons.append({'reference':name,'path':str(path),'file_sha256':sha(path),'equal_pixels':equal,'total_pixels':len(raw),'label_agreement':equal/len(raw),'different_pixels':len(raw)-equal})
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'labels_native.uint8').write_bytes(raw)
    report={'bundle':str(a.bundle),'execution_kind':bundle.run_spec.get('execution_kind'),'item_index':item,'shape':[h,w],'output_sha256':hashlib.sha256(raw).hexdigest(),'label_min':min(raw),'label_max':max(raw),'comparisons':comparisons,'scope':'single real image, diagnostic direct graph; not val500 task quality or steady-state performance'}
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
