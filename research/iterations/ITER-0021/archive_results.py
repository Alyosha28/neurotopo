"""Immutable two-copy archive with streaming SHA-256 verification."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);a=p.parse_args()
if a.destination.exists() or a.inventory.exists():raise FileExistsError('Archive is immutable')
a.destination.mkdir(parents=True);files={}
for src in sorted(a.source.rglob('*')):
 if not src.is_file() or src.name.startswith('._'):continue
 relative=src.relative_to(a.source);dst=a.destination/relative;dst.parent.mkdir(parents=True,exist_ok=True)
 h=hashlib.sha256()
 with src.open('rb') as f,dst.open('xb') as g:
  while chunk:=f.read(1024*1024):h.update(chunk);g.write(chunk)
 check=hashlib.sha256()
 with dst.open('rb') as f:
  while chunk:=f.read(1024*1024):check.update(chunk)
 if h.hexdigest()!=check.hexdigest():raise RuntimeError('Copy mismatch '+str(relative))
 files[str(relative)]={'sha256':h.hexdigest(),'bytes':src.stat().st_size}
value={'source':str(a.source),'project_copy':str(a.destination),'file_count':len(files),'bytes':sum(x['bytes'] for x in files.values()),'all_hashes_equal':True,'files':files}
with a.inventory.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({k:value[k] for k in ('file_count','bytes','all_hashes_equal')}))
