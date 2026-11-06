"""Run the authorized two-arm pilot after numerical and distributed checks."""
from datetime import datetime,timezone,timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/h20_motion_gate_20261031_01'
START=ROOT/'experiments/h20_optimization_20261030_01/run_A_differential/epoch_09.pth'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'ENV/h20_training_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    python=str(ROOT/'ENV/bin/python');started=time.monotonic();timings=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',CVCI_PREFETCH_FACTOR='2',PYTHONUNBUFFERED='1',NCCL_DEBUG='WARN',
        CVCI_RECORD_INPUT_HASHES='1',CVCI_WORLD_READOUT_STEPS='5')
    def status(phase,**details):
        state={'phase':phase,'pid':os.getpid(),'output_root':str(OUT),'elapsed_seconds':time.monotonic()-started,
            'updated_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'phase_timings':timings,**details}
        target=OUT/'state.json';temporary=OUT/'state.json.tmp';temporary.write_text(json.dumps(state,indent=2));os.replace(temporary,target)
        (ROOT/'ENV/h20_motion_gate_latest.json').write_text(json.dumps({'output_root':str(OUT),'state':str(target),'phase':phase},indent=2))
        print('MOTION_GATE_STATE',json.dumps(state),flush=True)
    def sources_unchanged():
        manifest=json.loads((OUT/'installation_manifest.json').read_text())
        for record in manifest['files']:
            assert hashlib.sha256((ROOT/record['name']).read_bytes()).hexdigest()==record['after_sha256'],record['name']
    def idle():
        busy=subprocess.getoutput('nvidia-smi --query-compute-apps=pid --format=csv,noheader').strip()
        assert not busy,'Other GPU processes active: '+busy
    def run(command,log,phase,deterministic=False):
        sources_unchanged();idle();phase_start=time.monotonic()
        selected=dict(env)
        if deterministic:selected.update(CVCI_DETERMINISTIC_CHECKS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NCCL_ALGO='Ring',NCCL_PROTO='Simple')
        log.parent.mkdir(parents=True,exist_ok=True)
        assert not log.exists(),'Refusing to overwrite prior phase log: '+str(log)
        with log.open('w') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=selected,stdout=stream,stderr=subprocess.STDOUT)
            status(phase,active_pid=child.pid,log=str(log),command=command)
            code=child.wait()
        timings.append({'phase':phase,'seconds':time.monotonic()-phase_start,'exit_code':code})
        if code:raise RuntimeError(phase+' failed; see '+str(log))
    def train(work,motion=False,smoke=False,resume=False,stop=0):
        command=[python,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(ROOT/'tools/train_h20.py'),
            '--work-dir',str(work),'--data-root','/mnt/workspace/dataSet/nuscene','--world-root',str(ROOT/'world_configs'),
            '--batch-size','12','--grad-accum','1','--workers','2' if smoke else '12','--epochs','2','--precision','bf16',
            '--lr','1e-4','--optimizer-profile','differential','--loss-diagnostics','--world-readout-steps','5',
            '--weight-decay','1e-4','--ddim-steps','1','--points-per-map','36','25','16','9','9','--window-m','6','3',
            '--offset-mode','learned','--future-dropout','0.1','--monitor-every','4' if smoke else '100',
            '--log-every','2' if smoke else '20','--verify-ddp','--seed','2026','--warmup-steps','2' if smoke else '100']
        if smoke:command+=['--steps-per-epoch','4']
        if motion:command+=['--motion-state-gate']
        if resume:command+=['--resume',str(work/'last.pth')]
        else:command+=['--warm-start',str(START)]
        if stop:command+=['--stop-after-epoch',str(stop)]
        phase=('smoke_' if smoke else 'training_')+('gate' if motion else 'control')+('_resume' if resume else '')
        run(command,work/('train_resume.log' if resume else 'train.log'),phase,deterministic=smoke)
    def evaluate(checkpoint,directory,mode='future'):
        command=[python,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            str(ROOT/'tools/eval_h20_protocol.py'),'--mode',mode,'--adapter-checkpoint',str(checkpoint),
            '--output-dir',str(directory),'--seed','2026','--ddim-steps','1','--ddim-schedule','pretrained',
            '--workers','6','--prefetch-factor','2','--log-every','40','--log-seconds','30']
        run(command,directory.parent/(directory.name+'.log'),'evaluating_'+directory.name)
        metrics=json.loads((directory/'metrics.json').read_text())
        assert metrics['sample_coverage_check']=='PASS' and metrics['complete_samples']==4585 and not metrics['partial']
        assert metrics['noise_sha256']=='334c41f7dd01a5f3afae130a7a4c42b4902969feef058ff873e52b24f8972e37'
        assert metrics['expected_complete_token_sha256']=='fd8fa4bc6f00938f4a1975ea499cead772670275c299877826eb23674f4b4380'
    try:
        status('preflight');assert shutil.disk_usage(OUT).free>24*1024**3
        sources_unchanged();assert START.is_file()
        verification=OUT/'checks/verification.json'
        if not verification.exists():
            status('waiting_for_numerical_checks')
            pid=json.loads((OUT/'precheck_state.json').read_text())['pid']
            for _ in range(300):
                if verification.exists():break
                assert Path('/proc/'+str(pid)).exists(),'Numerical check process exited without PASS'
                time.sleep(2)
        assert json.loads(verification.read_text())['status']=='PASS'
        for _ in range(30):
            if not subprocess.getoutput('nvidia-smi --query-compute-apps=pid --format=csv,noheader').strip():break
            time.sleep(1)
        train(OUT/'checks/smoke_control',smoke=True,stop=1)
        train(OUT/'checks/smoke_gate',motion=True,smoke=True,stop=1)
        train(OUT/'checks/smoke_gate',motion=True,smoke=True,resume=True)
        audit=json.loads((OUT/'checks/smoke_gate/resume_state_audit.json').read_text());assert audit['status']=='PASS'
        status('distributed_checks_passed',resume_audit=str(OUT/'checks/smoke_gate/resume_state_audit.json'))
        train(OUT/'run_control')
        evaluate(OUT/'run_control/epoch_02.pth',OUT/'validation/control_epoch02')
        train(OUT/'run_gate',motion=True)
        evaluate(OUT/'run_gate/epoch_02.pth',OUT/'validation/gate_epoch02')
        evaluate(OUT/'run_gate/epoch_02.pth',OUT/'validation/gate_epoch02_disabled',mode='disabled')
        run([python,'-u',str(ROOT/'tools/analyze_motion_gate_experiment.py')],OUT/'analysis.log','analyzing')
        result=json.loads((OUT/'comparison.json').read_text());assert result['status']=='PASS'
        status('complete',results=str(OUT/'comparison.json'),report=str(OUT/'状态门控小规模实验报告.md'),
            control_prefix_m=result['metrics']['control']['avg_L2_m'],motion_prefix_m=result['metrics']['motion_gate']['avg_L2_m'])
    except Exception:
        status('failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
