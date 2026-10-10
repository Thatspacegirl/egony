"""sha256 of every file under the given paths (relative to ROOT) -> JSON {relpath: sha256}.  Used to compare outputs
re-computed after the 2026-10-09 crash with the ones written in the crash window.
  python3 t3_hashdirs.py ROOT OUT.json PATH [PATH ...]
"""
import hashlib, os, sys, json
root=sys.argv[1]; out=sys.argv[2]; paths=sys.argv[3:]
res={}
for p in paths:
    full=os.path.join(root,p)
    if os.path.isfile(full):
        res[p]=hashlib.sha256(open(full,'rb').read()).hexdigest(); continue
    for dp,dn,fn in os.walk(full):
        dn.sort()
        for f in sorted(fn):
            q=os.path.join(dp,f); res[os.path.relpath(q,root)]=hashlib.sha256(open(q,'rb').read()).hexdigest()
json.dump(res,open(out+'.tmp','w'),indent=0); os.replace(out+'.tmp',out); print(len(res),'files hashed ->',out)
