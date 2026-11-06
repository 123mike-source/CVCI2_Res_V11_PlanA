"""Server-only paired analysis of the two equally budgeted motion-gate runs."""
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/h20_motion_gate_20261031_01'


def read(path):return json.loads(path.read_text())


def load_samples(directory):
    metrics=read(directory/'metrics.json');samples={}
    for rank in range(4):
        for line in (directory/('samples-rank%d.jsonl'%rank)).open():
            row=json.loads(line);assert row['token'] not in samples;samples[row['token']]=row
    assert len(samples)==5456 and sum(x['complete'] for x in samples.values())==4585
    assert metrics['sample_coverage_check']=='PASS' and metrics['complete_samples']==4585
    return metrics,samples


def frame_scores(samples,tokens):
    errors=np.asarray([samples[t]['errors'] for t in tokens],dtype=np.float64)
    prefix=np.stack([errors[:,:n].mean(1) for n in [2,4,6]],1).mean(1)
    return prefix,errors


def bootstrap(first,second,samples,tokens):
    scenes=sorted({samples[t]['scene'] for t in tokens});indices={s:[] for s in scenes}
    for i,t in enumerate(tokens):indices[samples[t]['scene']].append(i)
    sums=np.asarray([(first[indices[s]]-second[indices[s]]).sum() for s in scenes])
    counts=np.asarray([len(indices[s]) for s in scenes]);rng=np.random.RandomState(20261031)
    draw=rng.randint(len(scenes),size=(10000,len(scenes)))
    boot=sums[draw].sum(1)/counts[draw].sum(1)
    return {'first_minus_second_m':float((first-second).mean()),
        'ci95_scene_bootstrap_m':np.quantile(boot,[.025,.975]).tolist(),
        'scenes':len(scenes),'samples':len(tokens),'repeats':10000,
        'limitation':'Fixed checkpoints/noise and one training seed; post-hoc subgroups, no multiplicity adjustment.'}


def training_rows(work):
    rows=[]
    for line in (work/'train.log').open():
        try:x=json.loads(line)
        except ValueError:continue
        if 'loss_global' in x and x['step']==x['steps']:rows.append(x)
    assert len(rows)==2 and rows[-1]['optimizer_updates']==1174
    return rows


def main():
    paths={'control':OUT/'validation/control_epoch02','motion_gate':OUT/'validation/gate_epoch02',
           'motion_gate_disabled':OUT/'validation/gate_epoch02_disabled'}
    runs={k:load_samples(v) for k,v in paths.items()};reference=runs['control'][0];refrows=runs['control'][1]
    tokens=sorted(t for t,r in refrows.items() if r['complete'])
    for name,(metrics,samples) in runs.items():
        assert set(samples)==set(refrows)
        for field in ['noise_sha256','expected_complete_token_sha256','ddim_indices','metric_filter']:
            assert metrics[field]==reference[field]
        for t in tokens:assert samples[t]['scene']==refrows[t]['scene']
    assert runs['motion_gate'][0]['adapter_sha256']==runs['motion_gate_disabled'][0]['adapter_sha256']
    with open('/mnt/workspace/dataSet/nuscene/nuscenes_infos_val.pkl','rb') as stream:infos=pickle.load(stream)['infos']
    infos={str(x['token']):x for x in infos};speeds=[];distances=[];turns=[];max_discrepancy=0.
    for token in tokens:
        info=infos[token];assert str(info['scene_token'])==refrows[token]['scene']
        delta=np.asarray(info['gt_ego_fut_trajs'],dtype=np.float32)[:6,:2];truth=delta.cumsum(0)
        pred=np.asarray(refrows[token]['prediction_lidar'],dtype=np.float32)
        errors=np.sqrt(((pred-truth)**2).sum(-1));max_discrepancy=max(max_discrepancy,float(np.abs(errors-refrows[token]['errors']).max()))
        speeds.append(float(np.linalg.norm(delta[0])/.5));distance=float(np.linalg.norm(delta,axis=1).sum());distances.append(distance)
        a,b=delta[:2].sum(0),delta[-2:].sum(0)
        angle=abs(np.degrees(np.arctan2(a[0]*b[1]-a[1]*b[0],np.dot(a,b))))
        turns.append(distance>=3. and angle>=15.)
    assert max_discrepancy<1e-4
    speed=np.asarray(speeds);distance=np.asarray(distances)
    masks={'all':np.ones(len(tokens),dtype=bool),'low_start_speed':speed<=.5,
        'near_stationary':(speed<=.5)&(distance<=.5),'low_speed_starting':(speed<=.5)&(distance>=3.),
        'speed_0.5_to_5_mps':(speed>.5)&(speed<5.),'speed_ge_5_mps':speed>=5.,'turn_ge_15deg':np.asarray(turns)}
    assert masks['near_stationary'].sum()==611 and masks['low_speed_starting'].sum()==80 and masks['turn_ge_15deg'].sum()==371
    score={name:frame_scores(samples,tokens) for name,(metrics,samples) in runs.items()}
    groups={};paired={};csvrows=[]
    for label,mask in masks.items():
        group={'samples':int(mask.sum()),'models':{}}
        for name,(prefix,errors) in score.items():
            group['models'][name]={'prefix_avg_m':float(prefix[mask].mean()),
                'endpoint_3s_m':float(errors[mask,5].mean()),'prefix_p90_m':float(np.quantile(prefix[mask],.9)),
                'endpoint_3s_p90_m':float(np.quantile(errors[mask,5],.9))}
            csvrows.append({'group':label,'model':name,'samples':int(mask.sum()),**group['models'][name]})
        sub=[t for t,m in zip(tokens,mask) if m]
        paired[label]=bootstrap(score['motion_gate'][0][mask],score['control'][0][mask],refrows,sub)
        groups[label]=group
    configs={name:read(OUT/('run_'+name)/'config.json') for name in ['control','gate']}
    for cfg in configs.values():
        args=cfg['config'];assert args['t99_fraction']==0. and args['auxiliary_weights']==[.2,.2,.1]
        assert args['offset_lr'] is None and args['world_readout_steps']==5
        assert args['batch_size']==12 and cfg['nominal_effective_batch']==48
        assert cfg['lr_schedule']['total_updates']==1174 and cfg['lr_schedule']['warmup_updates']==100
    assert configs['control']['warm_start_audit']['source_sha256']==configs['gate']['warm_start_audit']['source_sha256']
    for epoch in [1,2]:
        assert read(OUT/'run_control'/('input_hash_epoch_%02d.json'%epoch))==read(OUT/'run_gate'/('input_hash_epoch_%02d.json'%epoch))
    training={name:training_rows(OUT/('run_'+name)) for name in ['control','gate']}
    saved=torch.load(OUT/'run_gate/epoch_02.pth',map_location='cpu')
    norms={n:float(v.float().norm()) for n,v in saved['trainable'].items() if n.endswith('.future_motion_projection.weight')}
    assert len(norms)==6 and all(np.isfinite(v) and v>0 for v in norms.values())
    gain=1.-runs['motion_gate'][0]['avg_L2_m']/runs['motion_gate_disabled'][0]['avg_L2_m']
    result={'status':'PASS','completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'protocol_check':'PASS','training_budget_and_inputs_check':'PASS','training_seed':2026,
        'source_dirs':{k:str(v) for k,v in paths.items()},'metrics':{k:v[0] for k,v in runs.items()},
        'groups':groups,'paired_motion_minus_control':paired,'training':training,
        'projection_weight_norms':norms,'motion_model_future_relative_gain':gain,
        'GT_error_reconstruction_max_difference_m':max_discrepancy,
        'group_definitions':'GT first displacement/.5s; near stationary: start<=.5m/s,path<=.5m; starting: start<=.5m/s,path>=3m; '
                            'turn: path>=3m and first/last two-displacement vectors angle>=15deg. Analysis only, never gate input.',
        'limitations':'Two additional epochs from A9, one training seed, fixed final epoch. Short controlled screening, not a final performance claim.'}
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    control=reference['avg_L2_m'];motion=runs['motion_gate'][0]['avg_L2_m'];ci=paired['all']['ci95_scene_bootstrap_m']
    report=['# 状态门控小规模对照实验','', '完成时间：'+result['completed_at'],'',
        '两组均从 A9 完整权重开始，额外训练 2 轮 / 1,174 次更新；优化器、历史和学习率日程重新初始化。'
        '唯一模型因素为六个零初始化 20→128 状态投影，共 15,360 个参数。B、E 与偏移学习率未启用改动。','',
        '| 指标 | A9 继续训练对照 | 新状态门控 |','|---|---:|---:|',
        '| prefix 平均 L2（m） | %.6f | %.6f |'%(control,motion),
        '| ego endpoint 平均 L2（m） | %.6f | %.6f |'%(reference['validation_planning_avg_L2_m'],runs['motion_gate'][0]['validation_planning_avg_L2_m']),
        '', '新门控相对对照的 prefix 变化为 %+.3f%%；新门控−对照的场景配对 95%% 区间为 [%.6f, %.6f] m。'%(100*(motion/control-1.),ci[0],ci[1]),
        '该区间固定权重与推理噪声，不涵盖训练种子变化。两轮短对照只能提供筛选信号。','',
        '| 事后分组 | 样本数 | 对照 prefix L2 | 新门控 prefix L2 | 对照 3s endpoint | 新门控 3s endpoint |',
        '|---|---:|---:|---:|---:|---:|']
    for label in ['near_stationary','low_speed_starting','turn_ge_15deg','speed_ge_5_mps']:
        group=groups[label];a=group['models']['control'];b=group['models']['motion_gate']
        report.append('| %s | %d | %.6f | %.6f | %.6f | %.6f |'%(label,group['samples'],a['prefix_avg_m'],b['prefix_avg_m'],a['endpoint_3s_m'],b['endpoint_3s_m']))
    report+=['','分组只用于统计，不能使用未来 GT 判断部署门控。3s endpoint 为完整六点样本的 LiDAR 单点误差，'
        '与上表各自掩码的 ego endpoint 平均值不是同一个指标。','',
        '新门控模型关闭未来分支时的 prefix L2 为 %.6f m，开启的相对收益为 %.3f%%。滚动历史随开关改变，属于整体系统对照。'%(runs['motion_gate_disabled'][0]['avg_L2_m'],100*gain),'',
        '两组每轮首批输入哈希一致，均使用全局 batch48、四卡 H20、BF16 可训练分支与 FP32 冻结 IRWM。'
        'warmup100更新，余弦降至各组峰值的 1e-3，保持 A 的峰值分组学习率。完整验证覆盖 6,019 帧、5,456 个规划样本、4,585 个六点有效样本，噪声和 token 哈希一致。','',
        '已有检查包括 FP32/BF16 零初始化等价、新参数梯度、未来分支屏蔽、v6/v7 权重兼容、四卡训练与恢复。'
        '本次不延长训练、不替换 A9 或旧实验最佳权重。','',
        '证据：comparison.json、comparison.csv、checks/verification.json、checks/smoke_gate/resume_state_audit.json、'
        'run_control/、run_gate/、validation/、installation_manifest.json、implementation.patch。', '']
    (OUT/'状态门控小规模实验报告.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({'status':'PASS','control_prefix_m':control,'motion_prefix_m':motion,
        'change_percent':100*(motion/control-1.),'ci95_m':ci,'future_gain_percent':100*gain,
        'report':str(OUT/'状态门控小规模实验报告.md')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
