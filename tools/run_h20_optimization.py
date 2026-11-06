"""Validate execution changes, then run the authorized differential-LR experiment."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',required=True)
    parser.add_argument('--baseline-work-dir',required=True);parser.add_argument('--checks-only',action='store_true')
    args=parser.parse_args();out=Path(args.output_root);out.mkdir(parents=True,exist_ok=True)
    baseline=Path(args.baseline_work_dir);started=time.monotonic()
    lock=(ROOT/'ENV/h20_training_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    checks=out/('checks_'+stamp);checks.mkdir();state=out/'state.json'
    python=str(ROOT/'ENV/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',CVCI_PREFETCH_FACTOR='2',PYTHONUNBUFFERED='1',NCCL_DEBUG='WARN')
    def status(phase,**details):
        value=dict(phase=phase,pid=os.getpid(),checks_dir=str(checks),output_root=str(out),
            elapsed_seconds=time.monotonic()-started,
            updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),**details)
        atomic_json(state,value);print('OPTIMIZATION_STATE',json.dumps(value),flush=True)
    def run(command,log,phase,single_gpu=False,deterministic_checks=False):
        status(phase,command=command,log=str(log))
        selected_env=dict(env,CUDA_VISIBLE_DEVICES='0') if single_gpu else dict(env)
        if deterministic_checks:
            selected_env.update(CVCI_DETERMINISTIC_CHECKS='1',PYTHONHASHSEED='0',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',NCCL_ALGO='Ring',NCCL_PROTO='Simple')
        with Path(log).open('wb',buffering=0) as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=selected_env,stdout=stream,stderr=subprocess.STDOUT)
            status(phase,active_pid=child.pid,command=command,log=str(log))
            code=child.wait()
        if code:raise RuntimeError('%s exited %s; see %s'%(phase,code,log))
    def training(work,epochs=10,steps=0,stop=0,resume=False,smoke=False):
        command=[python,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            str(ROOT/'tools/train_h20.py'),'--work-dir',str(work),'--data-root','/mnt/workspace/dataSet/nuscene',
            '--world-root',str(ROOT/'world_configs'),'--batch-size','12','--grad-accum','1',
            '--workers','2' if smoke else '12','--epochs',str(epochs),'--precision','bf16',
            '--lr','1e-4','--optimizer-profile','differential','--loss-diagnostics',
            '--world-readout-steps','5','--weight-decay','1e-4','--ddim-steps','1',
            '--points-per-map','36','25','16','9','9','--window-m','6','3',
            '--offset-mode','learned','--future-dropout','0.1','--monitor-every','4' if smoke else '100',
            '--log-every','2' if smoke else '10','--verify-ddp','--stop-after-epoch',str(stop)]
        if steps:command+=['--steps-per-epoch',str(steps),'--warmup-steps','4']
        if resume:command+=['--resume',str(work/'last.pth')]
        work.mkdir(parents=True,exist_ok=True)
        log=work/('train_resume.log' if resume else 'train.log')
        run(command,log,'smoke_training' if smoke else 'training_A',deterministic_checks=smoke)
    def evaluate(checkpoint,destination):
        command=[python,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            str(ROOT/'tools/eval_h20_protocol.py'),'--mode','future','--adapter-checkpoint',str(checkpoint),
            '--output-dir',str(destination),'--seed','2026','--ddim-steps','1','--ddim-schedule','pretrained',
            '--workers','6','--prefetch-factor','2','--log-every','20','--log-seconds','20']
        env['CVCI_WORLD_READOUT_STEPS']='5'
        run(command,destination.parent/(destination.name+'.log'),'evaluating_A')
        metrics=json.loads((destination/'metrics.json').read_text())
        assert metrics['sample_coverage_check']=='PASS' and not metrics['partial']
        assert metrics['seen_frames']==6019 and metrics['complete_samples']==4585
        assert metrics['noise_sha256']=='334c41f7dd01a5f3afae130a7a4c42b4902969feef058ff873e52b24f8972e37'
        assert metrics['expected_complete_token_sha256']=='fd8fa4bc6f00938f4a1975ea499cead772670275c299877826eb23674f4b4380'
        return metrics
    try:
        status('preflight')
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if busy:raise RuntimeError('GPU processes are already active: '+busy)
        assert shutil.disk_usage(out).free>12*1024**3,'Need 12 GiB free for independent test/training checkpoints'
        run([python,'-u',str(ROOT/'tools/validate_h20_optimization.py'),'--mode','policy',
            '--baseline-source',str(out/'source_before'),'--output',str(checks/'policy.json')],checks/'policy.log','checking_policy',True)
        run([python,'-u',str(ROOT/'tools/validate_h20_optimization.py'),'--mode','world',
            '--checkpoint',str(baseline/'epoch_08.pth'),'--output',str(checks/'world_equivalence.json')],
            checks/'world_equivalence.log','checking_world_readout',True)
        assert json.loads((checks/'world_equivalence.json').read_text())['status']=='PASS'
        training(checks/'smoke_resumed',epochs=2,steps=4,stop=1,smoke=True)
        training(checks/'smoke_resumed',epochs=2,steps=4,resume=True,smoke=True)
        import torch
        resumed=torch.load(str(checks/'smoke_resumed/last.pth'),map_location='cpu')
        audit=json.loads((checks/'smoke_resumed/resume_state_audit.json').read_text())
        assert audit['status']=='PASS' and len(audit['ranks'])==4
        assert all(all(r['checks'].values()) for r in audit['ranks'])
        assert resumed['optimizer_updates']==8 and resumed['next_epoch']==2
        agreement=[]
        for name in ('train.log','train_resume.log'):
            for line in (checks/'smoke_resumed'/name).read_text(errors='replace').splitlines():
                try:value=json.loads(line)
                except ValueError:continue
                if value.get('event')=='ddp_joint_agreement':agreement.append(value)
        assert {r['epoch'] for r in agreement}=={1,2} and all(r['ranks']==4 for r in agreement)
        input_reference=out/'validation_input_reference.json';input_checked=False
        if input_reference.exists():
            ref=json.loads(input_reference.read_text())
            assert ref['dataset_stream_sha256']==resumed['metadata']['dataset_stream_sha256']
            assert all(resumed['metadata']['code_sha256'][k]==v for k,v in ref['data_source_sha256'].items())
            for epoch in (1,2):
                observed=json.loads((checks/'smoke_resumed'/('input_hash_epoch_%02d.json'%epoch)).read_text())
                assert observed==ref['hashes'][str(epoch)],'Restored execution received different input batches'
            input_checked=True
        atomic_json(checks/'resume_equivalence.json',dict(status='PASS',exact_restored_states=audit,
            completed_updates=8,ddp_agreement=agreement,input_reference_checked=input_checked,
            numerical_trajectory_equality='not asserted: CUDA grid_sample backward is nondeterministic',
            prior_independent_run_diagnosis=str(out/'controlled_resume_difference.json')))
        del resumed
        run([python,'-u',str(ROOT/'tools/probe_h20_task_gradients.py'),'--checkpoint',str(baseline/'epoch_08.pth'),
            '--output',str(checks/'task_gradients.json')],checks/'task_gradients.log','probing_task_gradients',True)
        atomic_json(checks/'verification.json',dict(status='PASS',policy=True,world_readout=True,ddp=True,resume=True,task_probe=True))
        if args.checks_only:status('checks_complete');return
        work=out/'run_A_differential';evaluation=work/'validation';evaluation.mkdir(parents=True)
        assert not list(work.glob('epoch_*.pth')),'Refuse to overwrite an existing experiment A'
        training(work,stop=2)
        baseline_rows={r['epoch']:r for r in json.loads((baseline/'validation_20261030_110614/all_epochs.json').read_text())['results']}
        rows=[]
        def collect(epoch):
            metrics=evaluate(work/('epoch_%02d.pth'%epoch),evaluation/('epoch_%02d'%epoch))
            reference=baseline_rows[epoch]
            row=dict(epoch=epoch,prefix_avg_m=metrics['avg_L2_m'],endpoint_avg_m=metrics['validation_planning_avg_L2_m'],
                endpoint_3s_m=metrics['validation_planning_L2_m'][2],baseline_prefix_avg_m=reference['trajectory_L2_avg_m'],
                prefix_reduction_fraction=1-metrics['avg_L2_m']/reference['trajectory_L2_avg_m'])
            rows.append(row);atomic_json(evaluation/'comparison.json',dict(complete=len(rows)==10,results=rows))
            with (evaluation/'comparison.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
            status('epoch_evaluated',latest_result=row,completed_epochs=[r['epoch'] for r in rows])
        collect(1);collect(2)
        atomic_json(out/'early_A_results.json',dict(status='PASS',results=rows,
            critical_options=dict(t99_fraction=0,auxiliary_weights=[.2,.2,.1],offset_lr=1e-4),
            pending_review='t99 mixture and auxiliary-weight experiments remain disabled'))
        training(work,resume=True)
        for epoch in range(3,11):collect(epoch)
        best_prefix=min(rows,key=lambda r:r['prefix_avg_m']);best_endpoint=min(rows,key=lambda r:r['endpoint_avg_m'])
        atomic_json(evaluation/'best.json',dict(best_prefix=best_prefix,best_endpoint=best_endpoint,
            original_best_prefix_m=.3660032708033625,original_best_endpoint_m=.7181426360356902))
        status('complete',best_prefix=best_prefix,best_endpoint=best_endpoint)
    except BaseException as error:
        status('failed',error=type(error).__name__+': '+str(error));traceback.print_exc();raise


if __name__=='__main__':main()
