"""Evaluate every checkpoint on full validation scenes and select three by endpoint L2."""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    work=Path(args.work_dir);output=Path(args.output_root)
    output.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'ENV/h20_evaluation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();current=None;rows=[]
    def status(phase,**details):
        value=dict(phase=phase,pid=os.getpid(),current_epoch=current,completed_epochs=[r['epoch'] for r in rows],
            elapsed_seconds=time.monotonic()-started,updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
            output_root=str(output),**details)
        atomic_json(output/'state.json',value);print('EVAL_STATE',json.dumps(value),flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',NCCL_DEBUG='WARN')
    def run(checkpoint,destination,smoke=False):
        command=[str(ROOT/'ENV/bin/python'),'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            str(ROOT/'tools/eval_h20_protocol.py'),'--mode','future','--adapter-checkpoint',str(checkpoint),
            '--output-dir',str(destination),'--seed','2026','--ddim-steps','1','--ddim-schedule','pretrained',
            '--workers',str(args.workers),'--prefetch-factor','2','--log-every','20','--log-seconds','20']
        if smoke:command+=['--max-scenes','4']
        with (output/(destination.name+'.log')).open('wb',buffering=0) as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/'active_process.json',dict(pid=process.pid,command=command,smoke=smoke))
            code=process.wait()
        if code:raise RuntimeError('Evaluation exited with code %d; see %s'%(code,output/(destination.name+'.log')))
        return json.loads((destination/'metrics.json').read_text())
    try:
        checkpoints=sorted(work.glob('epoch_[0-9][0-9].pth'))
        assert len(checkpoints)==10,'Expected all ten epoch checkpoints'
        status('smoke_test',split='val',selection_metric='validation_planning_avg_L2_m',smoke_scenes=4)
        smoke_path=output/'smoke'
        if not (smoke_path/'metrics.json').exists():
            if smoke_path.exists() and any(smoke_path.iterdir()):
                smoke_path.rename(output/('smoke_incomplete_'+str(int(time.time()))))
            smoke=run(checkpoints[-1],smoke_path,smoke=True)
        else:smoke=json.loads((smoke_path/'metrics.json').read_text())
        assert smoke['sample_coverage_check']=='PASS' and smoke['partial']
        reference=None
        for checkpoint in checkpoints:
            current=int(checkpoint.stem.split('_')[1]);destination=output/checkpoint.stem
            status('evaluating_epoch',checkpoint=str(checkpoint),split='val',full_validation_frames=6019)
            metrics_path=destination/'metrics.json'
            if metrics_path.exists():metrics=json.loads(metrics_path.read_text())
            else:
                if destination.exists() and any(destination.iterdir()):
                    destination.rename(output/(checkpoint.stem+'_incomplete_'+str(int(time.time()))))
                metrics=run(checkpoint,destination)
            assert not metrics['partial'] and metrics['sample_coverage_check']=='PASS'
            assert metrics['seen_frames']==6019 and metrics['adapter_epoch']==current
            assert metrics['adapter_sha256']==digest(checkpoint)
            assert metrics['seed']==2026 and metrics['ddim_indices']==[99]
            keys=['noise_sha256','expected_complete_token_sha256','expected_seen_frames',
                  'expected_planning_frames','expected_complete_samples','world_checkpoint',
                  'validation_planning_valid_counts','sampler_config']
            comparable={key:metrics[key] for key in keys}
            if reference is None:reference=comparable
            assert reference==comparable,'Evaluation coverage/randomness/configuration differs between epochs'
            endpoints=metrics['validation_planning_L2_m'];prefix=metrics['prefix_l2']
            row=dict(epoch=current,selection_endpoint_avg_m=metrics['validation_planning_avg_L2_m'],
                endpoint_1s_m=endpoints[0],endpoint_2s_m=endpoints[1],endpoint_3s_m=endpoints[2],
                trajectory_L2_1s_m=prefix[0],trajectory_L2_2s_m=prefix[1],trajectory_L2_3s_m=prefix[2],
                trajectory_L2_avg_m=metrics['avg_L2_m'],complete_samples=metrics['complete_samples'],
                planning_frames=metrics['planning_frames'],seen_frames=metrics['seen_frames'],
                checkpoint=str(checkpoint),checkpoint_sha256=metrics['adapter_sha256'],
                metrics=str(metrics_path))
            rows.append(row)
            atomic_json(output/'all_epochs.json',dict(split='val',selection_metric='validation_planning_avg_L2_m',
                complete=len(rows)==10,results=rows))
            with (output/'all_epochs.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
            status('epoch_complete',metrics=row)
        selected=sorted(rows,key=lambda row:(row['selection_endpoint_avg_m'],row['endpoint_3s_m'],row['epoch']))[:3]
        folder=output/'selected';folder.mkdir(exist_ok=True)
        for rank,row in enumerate(selected,1):
            row['rank']=rank
            link=folder/('rank_%d_epoch_%02d.pth'%(rank,row['epoch']))
            if not link.exists():link.symlink_to(os.path.relpath(row['checkpoint'],folder))
            assert link.resolve()==Path(row['checkpoint']).resolve()
            row['selected_checkpoint_link']=str(link)
        report=dict(status='PASS',split='val',evaluated_epochs=10,validation_frames_per_epoch=6019,
            selection_metric='mean of independently masked ego endpoint L2 at 1,2,3 seconds after cold start',
            protocol='cvci3_cold_start_3_frames',seed=2026,ddim_indices=[99],
            same_validation_split_used_for_selection_and_reporting=True,selected=selected,
            elapsed_seconds=time.monotonic()-started)
        atomic_json(folder/'top3.json',report)
        headers=['rank','epoch','selection_endpoint_avg_m','endpoint_1s_m','endpoint_2s_m','endpoint_3s_m',
            'trajectory_L2_1s_m','trajectory_L2_2s_m','trajectory_L2_3s_m','trajectory_L2_avg_m']
        lines=['# Full validation checkpoint selection','',
            'All ten epochs use the same 6,019 validation frames, scene histories, CUDA noise bank and one-step DDIM [99].',
            'Ranking uses the average ego endpoint L2 at 1/2/3 seconds with individual masks after cold start.',
            'Trajectory L2 uses the existing complete-six-point LiDAR protocol. These are validation results.','',
            '| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
        for row in selected:
            lines.append('| '+' | '.join(str(row[key]) if key in ['rank','epoch'] else '%.6f'%row[key] for key in headers)+' |')
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
        status('complete',selected_epochs=[row['epoch'] for row in selected],report=str(output/'REPORT.md'))
    except BaseException as error:
        status('failed',error=type(error).__name__+': '+str(error));traceback.print_exc();raise


if __name__=='__main__':main()
