#!/usr/bin/env python3
"""Separate upstream-baseline changes from the Eagle-N plugin; verify original files."""
from __future__ import annotations
import difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent
REVIEW=BASE/'eagle_n_integration_review';REF=BASE/'SDV-BMT-v0.1.2-reference';ORIG=BASE/'SDV-BMT'
IGNORE={'.git','.venv','__pycache__','.pytest_cache','build','runs'}
def listing(root):
 out={}
 for p in root.rglob('*'):
  rel=p.relative_to(root)
  if any(x in IGNORE or x.endswith('.egg-info') for x in rel.parts) or not p.is_file() or p.suffix=='.pyc':continue
  out[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return out
def compare(a,b):return {'added':sorted(b.keys()-a.keys()),'removed':sorted(a.keys()-b.keys()),'modified':sorted(k for k in a.keys()&b.keys() if a[k]!=b[k]),'unchanged_count':sum(a[k]==b[k] for k in a.keys()&b.keys())}
def main():
 REVIEW.mkdir(exist_ok=True);o=listing(ORIG);r=listing(REF);w=listing(ROOT)
 snap=json.loads((REVIEW/'ORIGINAL_SNAPSHOT.json').read_text())['files_sha256']
 changed=[k for k,h in snap.items() if o.get(k)!=h]
 if changed:raise RuntimeError('Original source changed since copy: '+repr(changed))
 one=compare(o,w);two=compare(r,w);up=compare(o,r)
 private=[x for x in two['added'] if x.startswith('docs/porting/')]
 added=[x for x in two['added'] if x not in private]
 result={'original':str(ORIG),'reference':str(REF),'working_copy':str(ROOT),'original_files_verified_unchanged':len(snap),
         'working_vs_original':one,'working_vs_v012':two,'upstream_v012_vs_original':up,'private_guides_copied_not_plugin':private,
         'plugin_added_files':added,'plugin_modified_shared_files':two['modified'],'ignored_paths':sorted(IGNORE)}
 (REVIEW/'FOLDER_COMPARISON.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['FOLDER DIFF — EAGLE-N WORKING COPY','Original unchanged: '+str(len(snap))+' hashed files','Compare plugin against: '+str(REF),'Working copy: '+str(ROOT),'',
        'SHARED FILES MODIFIED VS v0.1.2:']+two['modified']+['','NEW PLUGIN / REVIEW FILES:']+added+['','PRIVATE GUIDES COPIED FROM ORIGINAL (not new plugin code):']+private
 lines+=['','Upstream upgrade vs original: '+str(len(up['added']))+' added, '+str(len(up['modified']))+' modified, '+str(len(up['removed']))+' removed.',
         'Plugin vs v0.1.2: '+str(len(added))+' added, '+str(len(two['modified']))+' modified shared, '+str(len(two['removed']))+' removed.',
         'Ignore .git/.venv/native/build/cache/egg-info when using a visual folder comparison.']
 (REVIEW/'CHANGES_EAGLE_N.txt').write_text('\n'.join(lines)+'\n')
 patch=[]
 for n in two['modified']:
  patch+=difflib.unified_diff((REF/n).read_text().splitlines(True),(ROOT/n).read_text().splitlines(True),fromfile='v0.1.2/'+n,tofile='eagle-n/'+n)
 (REVIEW/'SHARED_GLUE_VS_V012.patch').write_text(''.join(patch))
 print(json.dumps({'original_unchanged_files':len(snap),'added_plugin_files':len(added),'modified_shared':two['modified'],'removed':two['removed']},indent=2))
if __name__=='__main__':main()
