"""Verify data, benchmark H20, and start one four-GPU v6 training run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_paths import DEFAULT_DATA_ROOT,resolve_path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work-dir');parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--wait-data',action='store_true');parser.add_argument('--benchmark-only',action='store_true')
    args=parser.parse_args()
    lock=(ROOT/'ENV/h20_training_launch.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another H20 launch coordinator is active')
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    work=Path(args.work_dir) if args.work_dir else ROOT/'experiments'/('h20_gate12_'+stamp)
    work.mkdir(parents=True,exist_ok=True);state=work/'launch_state.json'
    def status(phase,**details):
        value=dict(phase=phase,pid=os.getpid(),work_dir=str(work),updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),**details)
        temporary=state.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(state)
        print('LAUNCH_STATE',json.dumps(value),flush=True)
    data=Path(DEFAULT_DATA_ROOT)
    required=[data/'nuscenes_infos_train.pkl',data/'nuscenes_infos_val.pkl',data/'v1.0-trainval/sample.json',
        ROOT/'world_configs/projects/configs/e2e/MMO_MSO_with_plan_tightly_coupled.py',
        ROOT.parent/'Models/best_endpoint_1step.pth',ROOT.parent/'Models/MMO_MSO_with_plan_tightly_coupled.pth']
    while True:
        missing=[str(p) for p in required if not p.is_file()]
        if not missing:break
        status('waiting_inputs',missing=missing)
        if not args.wait_data:raise SystemExit('Required training inputs are missing')
        time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    prior=work/'batch_benchmark.json'
    if not prior.exists() or json.loads(prior.read_text()).get('status')!='PASS':
        status('benchmarking_real_irwm')
        with (work/'batch_benchmark.log').open('w') as log:
            subprocess.run([str(ROOT/'ENV/bin/python'),'-u',str(ROOT/'tools/benchmark_h20_training.py'),
                            '--output',str(prior)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    benchmark=json.loads((work/'batch_benchmark.json').read_text());assert benchmark['status']=='PASS'
    batch=benchmark['selected']['batch_size'];accum=benchmark['grad_accum']
    if args.benchmark_only:status('benchmark_complete',benchmark=benchmark);return
    marker_names=['v1.0-trainval%02d_blobs.tgz.done'%i for i in range(1,11)]
    while True:
        missing=[name for name in marker_names if not (data/'_extracted'/name).is_file()]
        if not missing:break
        status('waiting_dataset_extraction',pending_archives=missing,benchmark=benchmark['selected'])
        if not args.wait_data:raise SystemExit('Training archives are still extracting')
        time.sleep(60)
    status('checking_all_sensor_paths')
    paths=set()
    for name in ['nuscenes_infos_train.pkl','nuscenes_infos_val.pkl']:
        with (data/name).open('rb') as stream:payload=pickle.load(stream)
        infos=payload['infos'] if isinstance(payload,dict) else payload
        for info in infos:
            paths.add(resolve_path(data,info['lidar_path']))
            paths.update(resolve_path(data,c['data_path']) for c in info['cams'].values())
    def absent(path):
        p=Path(path)
        return path if not p.is_file() or p.stat().st_size==0 else None
    with ThreadPoolExecutor(max_workers=24) as pool:missing=[p for p in pool.map(absent,paths) if p]
    if missing:
        status('missing_sensor_files',count=len(missing),examples=missing[:30]);raise SystemExit('Full sensor path check failed')
    cpu=len(os.sched_getaffinity(0));workers=min(12,max(2,cpu//4-6));prefetch=2
    shm=shutil.disk_usage('/dev/shm').free
    while workers>2 and batch*16*1024**2*workers*prefetch*4>shm*.6:workers-=1
    if batch*16*1024**2*workers*prefetch*4>shm*.6:workers=0
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',CVCI_PREFETCH_FACTOR=str(prefetch),PYTHONUNBUFFERED='1')
    command=[str(ROOT/'ENV/bin/python'),'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(ROOT/'tools/train_h20.py'),'--work-dir',str(work),'--data-root',str(data),
             '--world-root',str(ROOT/'world_configs'),'--batch-size',str(batch),'--grad-accum',str(accum),
             '--workers',str(workers),'--epochs',str(args.epochs),'--precision','bf16','--lr','1e-4',
             '--weight-decay','1e-4','--ddim-steps','1','--points-per-map','36','25','16','9','9',
             '--window-m','6','3','--offset-mode','learned','--future-dropout','0.1',
             '--monitor-every','100','--log-every','10','--verify-ddp']
    with (work/'train.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status('training',training_pid=process.pid,command=command,batch_per_gpu=batch,grad_accum=accum,
               effective_global_batch=48,workers_per_gpu=workers,prefetch_factor=prefetch,checked_sensor_files=len(paths))
        code=process.wait()
    status('complete' if code==0 else 'failed',training_returncode=code)
    if code:raise SystemExit(code)


if __name__=='__main__':main()
