"""Fetch a scoped GitHub LFS download and verify both original training indices."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path('/mnt/workspace/dataSet/nuscene')
EXPECTED={
    'nuscenes_infos_train.pkl':(758819122,'5a2e5cace004a56a4924a76fafd604e13aacddd83ae324f1f87448332133c759'),
    'nuscenes_infos_val.pkl':(155647706,'91b69afa4a17e5ba68a270a79d3673bb49edb2ce3f5efd3691f593788453ff7b'),
}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb',buffering=8*1024**2) as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    request=json.load(sys.stdin)
    target=ROOT/'_archives/hpc_annotations_github.zip.part'
    config='url = '+json.dumps(request['url'])+'\n'
    config+=''.join('header = '+json.dumps(k+': '+v)+'\n' for k,v in (request.get('headers') or {}).items())
    print('GITHUB_ANNOTATION_DOWNLOAD_START',flush=True)
    if request.get('parallel'):
        from download_annotations_parallel import fetch
        fetch(request,target)
    else:
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','5',
            '--retry-delay','2','--connect-timeout','30','--continue-at','-',
            '--output',str(target),'--write-out','DOWNLOAD bytes=%{size_download} speed=%{speed_download}\n',
            '--config','-'],input=config.encode(),check=True)
    assert target.stat().st_size==369915018, 'Archive size mismatch'
    assert digest(target)=='d1ec8c689ea902c869aacec4790e95de5d3753ff8257e0ae94732c77aedab317','Archive SHA mismatch'
    verified=[]
    with zipfile.ZipFile(target) as archive:
        for name,(size,sha) in EXPECTED.items():
            destination=ROOT/name
            if destination.exists():
                assert destination.stat().st_size==size and digest(destination)==sha,'Existing index differs'
            else:
                part=ROOT/('.'+name+'.github.part')
                with archive.open(name) as source,part.open('wb',buffering=8*1024**2) as output:
                    for block in iter(lambda:source.read(8*1024**2),b''):output.write(block)
                assert part.stat().st_size==size and digest(part)==sha,'Index verification failed'
                os.link(part,destination);part.unlink()
            verified.append(dict(name=name,bytes=size,sha256=sha))
            print('INDEX_VERIFIED',name,size,flush=True)
    manifest=dict(status='PASS',source='private_github_lfs',repository='123mike-source/CVCI_ckpts',
        branch='h20-training-assets-20261029',files=verified,verified_at=time.time())
    (ROOT/'annotation_transfer_manifest.json').write_text(json.dumps(manifest,indent=2))
    target.unlink()
    chunks=ROOT/'_archives/hpc_annotations_github.chunks'
    if chunks.exists():
        import shutil
        assert chunks.resolve().parent==(ROOT/'_archives').resolve()
        shutil.rmtree(chunks)
    print('ANNOTATIONS_GITHUB_READY',flush=True)

if __name__=='__main__':main()
