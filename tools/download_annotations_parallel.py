"""Download a large scoped LFS object in verified HTTP byte ranges."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import subprocess
import time

SIZE=369915018
CHUNK=1024*1024

def fetch(request,target):
    folder=target.parent/'hpc_annotations_github.chunks'
    folder.mkdir(exist_ok=True)
    config='url = '+json.dumps(request['url'])+'\n'
    config+=''.join('header = '+json.dumps(k+': '+v)+'\n' for k,v in (request.get('headers') or {}).items())
    def one(start):
        stop=min(SIZE,start+CHUNK)-1
        part=folder/('%012d.part'%start)
        for attempt in range(120):
            have=part.stat().st_size if part.exists() else 0
            assert have<=stop-start+1
            if have==stop-start+1:return part
            scratch=part.with_suffix('.attempt')
            result=subprocess.run(['curl','--fail','--location','--silent','--show-error',
                '--ipv4','--http1.1','--connect-timeout','15','--max-time','120',
                '--speed-limit','8192','--speed-time','20','--range','%d-%d'%(start+have,stop),
                '--output',str(scratch),'--write-out','%{http_code}','--config','-'],
                input=config.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if result.stdout.strip()==b'206' and scratch.exists():
                count=scratch.stat().st_size
                assert count<=stop-start+1-have,'Unexpected range response size'
                with part.open('ab') as destination,scratch.open('rb') as source:shutil.copyfileobj(source,destination)
            elif result.stdout.strip() in (b'401',b'403'):
                raise RuntimeError('Scoped download URL expired or access was refused')
            elif result.stdout.strip()==b'200':
                raise RuntimeError('The download server ignored the required byte range')
            if scratch.exists():scratch.unlink()
            if result.returncode:time.sleep(min(5,attempt+1))
        raise RuntimeError('Byte range repeatedly failed at '+str(start))
    starts=list(range(0,SIZE,CHUNK));completed=0;clock=time.monotonic()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures={pool.submit(one,start):start for start in starts}
        for future in as_completed(futures):
            future.result();completed+=1
            if completed%8==0 or completed==len(starts):
                print('LFS_RANGES',completed,'/',len(starts),'elapsed_s',round(time.monotonic()-clock,1),flush=True)
    with target.open('wb',buffering=8*1024**2) as destination:
        for start in starts:
            with (folder/('%012d.part'%start)).open('rb') as source:shutil.copyfileobj(source,destination,8*1024**2)
    assert target.stat().st_size==SIZE
    # The caller checks the full SHA before accepting this archive.
