#!/usr/bin/env python
"""Build MMCV 1.7.2 for H20 (sm_90) using an ENV-local CUDA 11.8/GCC 11 SDK."""
from pathlib import Path
import concurrent.futures,hashlib,importlib.metadata as metadata,io,json,os,subprocess,sys,tarfile,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1];ENV=ROOT/'ENV';CUDA=ENV/'cuda-11.8';BUILD=ENV/'_build/mmcv-sm90';DOWNLOAD=BUILD/'downloads';NATIVE=ENV/'native';WHEELS=ENV/'native_wheels'
for p in [CUDA,BUILD,DOWNLOAD,WHEELS]:p.mkdir(parents=True,exist_ok=True)
MARKER=ENV/'mmcv_sm90_build.json'
if MARKER.exists() and metadata.version('mmcv-full')=='1.7.2':
    print('MMCV_SM90_BUILD_ALREADY_AVAILABLE',flush=True);raise SystemExit(0)
def run(cmd,**kwargs):
    print('RUN',' '.join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),check=True,**kwargs)
def fetch(url,dest,sha=None):
    if dest.exists() and (not sha or hashlib.sha256(dest.read_bytes()).hexdigest()==sha):return
    part=Path(str(dest)+'.part')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url,timeout=120) as source,open(part,'wb') as target:
                while True:
                    chunk=source.read(4*1024**2)
                    if not chunk:break
                    target.write(chunk)
            if sha and hashlib.sha256(part.read_bytes()).hexdigest()!=sha:raise RuntimeError('SHA256 mismatch: '+url)
            part.replace(dest);return
        except Exception:
            if attempt==3:raise
def read_range(url,start,end):
    urls=[url.replace('https://files.pythonhosted.org/packages/','https://mirrors.aliyun.com/pypi/packages/'),url]
    last=None
    for candidate in dict.fromkeys(urls):
        try:
            data=subprocess.check_output(['curl','--fail','--location','--silent','--show-error','--connect-timeout','20','--max-time','120','--retry','3','--retry-all-errors','--retry-delay','2','--max-filesize','2097152','--range','%d-%d'%(start,end),candidate])
            if len(data)!=end-start+1:raise RuntimeError('Incomplete SDK header range')
            return data
        except Exception as exc:last=exc
    raise RuntimeError('SDK header range download failed: '+str(last))
class RangedZip(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.cache={};self.chunk=1024**2
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset;return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos);out=[]
        while n>0:
            index=self.pos//self.chunk;start=index*self.chunk;end=min(self.size,start+self.chunk)-1
            if index not in self.cache:
                req=urllib.request.Request(self.url,headers={'Range':'bytes=%d-%d'%(start,end)})
                self.cache[index]=read_range(self.url,start,end)
                if len(self.cache[index])!=end-start+1:raise RuntimeError('Incomplete SDK range response')
            offset=self.pos-start;take=min(n,len(self.cache[index])-offset)
            if take<=0:raise RuntimeError('Invalid SDK range')
            out.append(self.cache[index][offset:offset+take]);self.pos+=take;n-=take
        return b''.join(out)
def sdk_headers(project,version):
    marker=CUDA/('headers-'+project+'.json')
    if marker.exists():return
    with urllib.request.urlopen('https://pypi.org/pypi/%s/%s/json'%(project,version),timeout=60) as f:release=json.load(f)
    item=next(x for x in release['urls'] if x['filename'].endswith('.whl') and 'manylinux' in x['filename'] and 'x86_64' in x['filename'])
    print('FETCH_SDK_HEADERS_ONLY',project,version,flush=True);count=0
    with zipfile.ZipFile(RangedZip(item['url'],item['size'])) as archive:
        for member in archive.infolist():
            if '/include/' not in member.filename or member.is_dir():continue
            relative=member.filename.split('/include/',1)[1]
            if '..' in Path(relative).parts:raise RuntimeError('Unsafe SDK header path')
            target=CUDA/'include'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(member));count+=1
        for member in archive.infolist():
            if 'license' in member.filename.lower() and not member.is_dir():
                target=CUDA/'licenses'/project/Path(member.filename).name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(member))
    marker.write_text(json.dumps({'project':project,'version':version,'source':item['url'],'header_count':count},indent=2))
    print('SDK_HEADERS_READY',project,count,flush=True)
base='https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/'
with urllib.request.urlopen(base+'repodata.json',timeout=60) as f:data=json.load(f)
names={'cuda-nvcc','cuda-cudart','cuda-cudart-dev','cuda-cccl','cuda-driver-dev'}
packages=[(name,pkg) for name,pkg in data['packages'].items() if pkg['name'] in names]
for name,pkg in packages:
    marker=CUDA/(name+'.extracted')
    if marker.exists():continue
    path=DOWNLOAD/name;print('FETCH_CUDA_SDK',name,pkg['size'],flush=True);fetch(base+name,path,pkg.get('sha256'))
    with tarfile.open(path) as archive:archive.extractall(CUDA)
    marker.touch()
if not (CUDA/'lib64').exists():(CUDA/'lib64').symlink_to('lib')
headers=[('nvidia-cublas-cu11','11.11.3.6'),('nvidia-cusolver-cu11','11.4.1.48'),('nvidia-cusparse-cu11','11.7.5.86'),('nvidia-curand-cu11','10.3.0.86'),('nvidia-cufft-cu11','10.9.0.58')]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda item:sdk_headers(*item),headers))
gxx=NATIVE/'usr/bin/g++-11';gcc=NATIVE/'usr/bin/gcc-11'
if not gxx.exists():
    cache=NATIVE/'packages';cache.mkdir(parents=True,exist_ok=True)
    run(['apt-get','download','gcc-11','g++-11','cpp-11','libgcc-11-dev','libstdc++-11-dev'],cwd=cache)
    for package in cache.glob('*.deb'):run(['dpkg-deb','--extract',package,NATIVE])
stdlib=NATIVE/'usr/lib/x86_64-linux-gnu/libstdc++.so.6'
if not stdlib.exists():stdlib.symlink_to(Path('/lib/x86_64-linux-gnu/libstdc++.so.6').resolve())
env=dict(os.environ,CUDA_HOME=str(CUDA),CC=str(gcc),CXX=str(gxx),CUDAHOSTCXX=str(gxx),MMCV_WITH_OPS='1',FORCE_CUDA='1',TORCH_CUDA_ARCH_LIST='9.0',MAX_JOBS='8',GCC_EXEC_PREFIX=str(NATIVE/'usr/lib/gcc')+'/',CPLUS_INCLUDE_PATH=str(NATIVE/'usr/include/c++/11')+':'+str(NATIVE/'usr/include/x86_64-linux-gnu/c++/11'))
env['PATH']=str(CUDA/'bin')+':'+str(NATIVE/'usr/bin')+':'+str(ENV/'bin')+':'+env.get('PATH','')
env['LD_LIBRARY_PATH']=str(ENV/'lib/python3.8/site-packages/torch/lib')+':'+str(NATIVE/'usr/lib/x86_64-linux-gnu')+':'+str(CUDA/'lib')+':'+str(CUDA/'nvvm/lib64')+':'+env.get('LD_LIBRARY_PATH','')
run([CUDA/'bin/nvcc','--version'],env=env);run([gxx,'--version'],env=env)
probe=BUILD/'compiler_probe.cpp';probe.write_text('#include <functional>\nint main(){std::function<int()> f=[](){return 0;};return f();}\n')
run([gxx,probe,'-o',BUILD/'compiler_probe'],env=env)
with urllib.request.urlopen('https://pypi.org/pypi/mmcv-full/1.7.2/json',timeout=60) as f:release=json.load(f)
item=next(x for x in release['urls'] if x['packagetype']=='sdist');source=DOWNLOAD/item['filename'];fetch(item['url'],source,item['digests']['sha256'])
src=BUILD/'mmcv-full-1.7.2'
if not src.exists():
    with tarfile.open(source) as archive:archive.extractall(BUILD)
run([sys.executable,'setup.py','bdist_wheel','--dist-dir',WHEELS],cwd=src,env=env)
wheel=next(WHEELS.glob('mmcv_full-1.7.2-cp38-*-linux_x86_64.whl'))
run([sys.executable,'-m','pip','install','--no-deps','--force-reinstall',wheel],env=env)
MARKER.write_text(json.dumps({'mmcv':'1.7.2','torch':metadata.version('torch'),'cuda':'11.8','architecture':'sm_90','wheel':str(wheel)},indent=2))
print('MMCV_SM90_NATIVE_BUILD_COMPLETE',flush=True)