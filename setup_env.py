"""Build the project-local Python 3.8/CUDA 11.8 environment used by CVCI2_Res."""
from pathlib import Path
import os,subprocess,sys,json,time
ROOT=Path(__file__).resolve().parent
ENV=ROOT/'ENV'
UV=ROOT/'.env-tools/bin/uv'
settings=dict(os.environ,UV_PYTHON_INSTALL_DIR=str(ROOT/'.python'),UV_PYTHON_BIN_DIR=str(ROOT/'.env-tools/bin'),UV_CACHE_DIR=str(ROOT/'.env-cache'),UV_NATIVE_TLS='true',UV_INDEX_URL='https://mirrors.aliyun.com/pypi/simple',UV_HTTP_TIMEOUT='600',UV_HTTP_RETRIES='6')
def run(args):
    print('RUN',' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=ROOT,env=settings,check=True)
def state(phase,error=None):
    (ROOT/'.env-setup-state.json').write_text(json.dumps({'phase':phase,'pid':os.getpid(),'time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'error':error},indent=2))
def configure_native_libraries():
    import subprocess
    native=ENV/'native';library=native/'usr/lib/x86_64-linux-gnu'
    if (library/'libGLX.so.0').exists() and (library/'libGLdispatch.so.0').exists():return
    cache=native/'packages';cache.mkdir(parents=True,exist_ok=True)
    print('Downloading ENV-local OpenCV GLX/GLdispatch dependencies',flush=True)
    subprocess.run(['apt-get','download','libglx0','libglvnd0'],cwd=cache,check=True)
    for package in cache.glob('libgl*.deb'):
        subprocess.run(['dpkg-deb','--extract',str(package),str(native)],check=True)
    if not (library/'libGLX.so.0').exists() or not (library/'libGLdispatch.so.0').exists():raise RuntimeError('ENV-local OpenCV native libraries are missing')
def configure_activation():
    import re
    path=ENV/'bin/activate';text=path.read_text()
    marker='# CVCI2_Res native library activation'
    if marker in text:return
    restore='''    # CVCI2_Res native library deactivation
    if [ "${_CVCI_LD_SAVED+x}" = x ]; then
        if [ "$_CVCI_LD_SAVED" = 1 ]; then
            export LD_LIBRARY_PATH="$_CVCI_OLD_LD_LIBRARY_PATH"
        else
            unset LD_LIBRARY_PATH
        fi
        unset _CVCI_LD_SAVED _CVCI_OLD_LD_LIBRARY_PATH
    fi
'''
    text,count=re.subn(r'(deactivate\s*\(\)\s*\{\n)',lambda m:m.group(1)+restore,text,count=1)
    if count!=1:raise RuntimeError('Unknown virtual environment activation format')
    text+='''
# CVCI2_Res native library activation
if [ "${LD_LIBRARY_PATH+x}" = x ]; then
    _CVCI_LD_SAVED=1
    _CVCI_OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
else
    _CVCI_LD_SAVED=0
    _CVCI_OLD_LD_LIBRARY_PATH=""
fi
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.8/site-packages/torch/lib:$VIRTUAL_ENV/native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
'''
    path.write_text(text)
def configure_library_loader():
    import re
    library=ENV/'lib/python3.8/site-packages/torch/lib'
    for file in library.iterdir():
        match=re.match(r'^(lib.+?)-[0-9a-f]+(\.so(?:\..*)?)$',file.name)
        if match:
            alias=library/(match.group(1)+match.group(2))
            if alias.is_symlink() and __import__("os").readlink(alias)!=file.name:alias.unlink()
            if not alias.exists():alias.symlink_to(file.name)
    python=ENV/'bin/python';real=ENV/'bin/python-real'
    marker='# CVCI2_Res isolated libtorch launcher'
    if python.is_symlink():
        target=python.resolve()
        if not real.exists():real.symlink_to(target)
        elif real.resolve()!=target:raise RuntimeError('Unexpected ENV python-real target')
        python.unlink()
    elif not python.exists() or marker not in python.read_text():
        raise RuntimeError('Unexpected existing ENV Python launcher')
    launcher='''#!/bin/sh
# CVCI2_Res isolated libtorch launcher
CVCI_ENV_BIN=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export LD_LIBRARY_PATH="$CVCI_ENV_BIN/../lib/python3.8/site-packages/torch/lib:$CVCI_ENV_BIN/../native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$CVCI_ENV_BIN/python-real" "$@"
'''
    python.write_text(launcher);python.chmod(0o755)
try:
    state('bootstrap')
    if not UV.is_file():run([sys.executable,'-m','pip','install','--target',ROOT/'.env-tools','--index-url','https://mirrors.aliyun.com/pypi/simple','uv==0.8.22'])
    state('python38')
    run([UV,'python','install','3.8.20'])
    candidates=list((ROOT/'.python').glob('cpython-3.8.20-*/bin/python3.8'))
    if len(candidates)!=1:raise RuntimeError('Managed Python 3.8 interpreter not found: '+repr(candidates))
    if not (ENV/'pyvenv.cfg').exists():
        if ENV.exists() and any(ENV.iterdir()):raise RuntimeError('Refusing to replace existing non-venv ENV')
        run([UV,'venv','--python',candidates[0],'--seed',ENV])
    else:
        version=subprocess.check_output([str(ENV/'bin/python'),'-c','import sys;print(sys.version_info[:2])'],text=True)
        if '(3, 8)' not in version:raise RuntimeError('Existing ENV uses an incompatible Python version')
    (ENV/'pip.conf').write_text('[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple\ntimeout = 120\n')
    state('installing_dependencies')
    run([UV,'pip','install','--python',ENV/'bin/python','--index-url','https://mirrors.aliyun.com/pypi/simple','-r',ROOT/'requirements-env.txt'])
    configure_native_libraries()
    configure_library_loader()
    configure_activation()
    state('building_mmcv_sm90')
    run([ENV/'bin/python',ROOT/'tools/build_mmcv_sm90.py'])
    state('checking_dependencies')
    run([ENV/'bin/python','-m','pip','check'])
    with open(ENV/'requirements.lock.txt','w') as f:subprocess.run([str(ENV/'bin/python'),'-m','pip','freeze'],stdout=f,env=settings,check=True)
    state('installed_pending_validation')
    print('CVCI_ENV_INSTALLATION_DONE',flush=True)
except BaseException as exc:
    state('failed',str(exc));raise