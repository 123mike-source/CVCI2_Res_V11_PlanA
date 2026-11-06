"""Benchmark real-size joint IRWM training on available consecutive real samples."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_paths import DEFAULT_DATA_ROOT,resolve_path
from temporal_dataset import TemporalNuScenesDataset,temporal_collate,to_device
from temporal_matched_deform_model import MatchedDeformTemporalModel
from train_h20 import configure_world_batch


def sequence(dataset):
    for indices in dataset.scenes.values():
        if len(indices)<7:continue
        chosen=indices[:7]
        infos=[dataset.infos[i] for i in chosen]
        if any(not .4 <= (b['timestamp']-a['timestamp'])/1e6 <= .6 for a,b in zip(infos,infos[1:])):continue
        paths=[resolve_path(dataset.data_root,i['lidar_path']) for i in infos]
        paths += [resolve_path(dataset.data_root,c['data_path']) for i in infos for c in i['cams'].values()]
        if all(Path(p).is_file() and Path(p).stat().st_size>0 for p in paths):
            try:return [dataset[i] for i in chosen]
            except FileNotFoundError:continue
    raise RuntimeError('No fully extracted consecutive 7-frame sequence is available yet')


def run(batch_size,frames):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model=MatchedDeformTemporalModel(fusion_config={'future_dropout':0.},sampler_config={
        'points_per_map':[36,25,16,9,9],'window_m':[6.,3.],'offset_mode':'learned'}).cuda().train()
    configure_world_batch(model,batch_size)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,fused=True)
    seconds=[];loads=[]
    for frame,sample in enumerate(frames):
        start=time.perf_counter()
        batch=to_device(temporal_collate([copy.deepcopy(sample) for _ in range(batch_size)]),torch.device('cuda'))
        for lane,meta in enumerate(batch['meta']):
            meta['scene_token']=str(meta['scene_token'])+':lane'+str(lane)
            meta['token']=str(meta['token'])+':lane'+str(lane)
        torch.cuda.synchronize();loads.append(time.perf_counter()-start)
        start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):result=model(batch,ddim_steps=1)
        result['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        assert torch.isfinite(result['loss'])
        if frame>=4:
            assert bool(result['future_used'].all());seconds.append(elapsed)
        print('BATCH_BENCH',batch_size,frame,round(elapsed,4),'future',int(result['future_used'].sum()),flush=True)
    row=dict(batch_size=batch_size,seconds_per_batch=sum(seconds)/len(seconds),
             samples_per_second=batch_size*len(seconds)/sum(seconds),
             peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
             peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
             host_collate_seconds=sum(loads)/len(loads),all_active_future=True,finite_gradients=True)
    del model,optimizer,result,batch;gc.collect();torch.cuda.empty_cache()
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch-sizes',nargs='+',type=int,default=[4,6,12])
    parser.add_argument('--annotations',default=str(Path(DEFAULT_DATA_ROOT)/'nuscenes_infos_train.pkl'))
    parser.add_argument('--resume-report',action='store_true')
    parser.add_argument('--output',default=str(ROOT/'ENV/h20_batch_benchmark.json'));args=parser.parse_args()
    torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
    dataset=TemporalNuScenesDataset(DEFAULT_DATA_ROOT,args.annotations)
    frames=sequence(dataset);rows=[]
    if args.resume_report and Path(args.output).exists():
        rows=json.loads(Path(args.output).read_text())['results']
    for size in args.batch_sizes:
        if any(row.get('batch_size')==size and 'samples_per_second' in row for row in rows):continue
        try:rows.append(run(size,frames))
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(batch_size=size,error='out_of_memory'));gc.collect();torch.cuda.empty_cache();break
        Path(args.output).write_text(json.dumps(dict(status='RUNNING',results=rows),indent=2))
    good=[x for x in rows if 'samples_per_second' in x]
    if not good:raise RuntimeError('No candidate batch completed')
    selected=max(good,key=lambda x:x['samples_per_second'])
    result=dict(status='PASS',results=rows,selected=selected,physical_gpus=4,
                effective_global_batch=48,grad_accum=48//(4*selected['batch_size']),
                benchmark_annotations=args.annotations,benchmark_records=len(dataset),
                note='Batch candidates preserve the original effective global batch of 48')
    Path(args.output).write_text(json.dumps(result,indent=2));print('H20_BATCH_BENCHMARK_PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
