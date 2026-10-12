"""从完整实测轨迹计算配对指标和逐完整驱动周期诊断。

周期用于描述时间结构，不是独立重复样本；世界前向/侧漂不旋转，
周期边界仅在实测角同相位穿越处线性定位，原始轨迹不平滑。
"""
from pathlib import Path
import json
import numpy as np


def upward_crossings(time,angle,level,refractory_s=0):
    """定位正向穿越，去掉同一过渡附近的抖动穿越，不制造新周期。"""
    indices=np.where((angle[:-1]<level)&(angle[1:]>=level))[0]
    result=[]
    for index in indices:
        alpha=(level-angle[index])/(angle[index+1]-angle[index])
        crossing=float(time[index]+alpha*(time[index+1]-time[index]))
        if not result or crossing-result[-1]>refractory_s: result.append(crossing)
    return result


def analyze_record(path):
    """正式完整轨迹才生成主指标，失败前缀不能冒充完整部署。"""
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['scope']=='formal_paired_deployment'
    data=np.genfromtxt(path/'trajectory.csv',delimiter=',',names=True)
    assert len(data)==manifest['requested_steps']==manifest['completed_steps']
    assert all(np.isfinite(data[name]).all() for name in data.dtype.names)
    primitive=manifest['primitive']
    time=data['time_s']
    if primitive=='S01':
        start=0
        origin=np.asarray(manifest['initial_root_com'][:2])
        window_start=0.
    else:
        from mjlab.tasks.stage2_skill.settling import track_settling,require_settled
        settling=track_settling(data['settled'].astype(bool),np.zeros(len(data),dtype=bool))
        require_settled(settling)
        start=int(settling.segment_start)
        origin=np.array([data['com_x'][start],data['com_y'][start]])
        window_start=float(time[start])
    sign=-1 if primitive=='S01' else 1
    xy=np.column_stack((data['com_x'],data['com_y']))
    displacement=xy[-1]-origin
    duration=float(time[-1]-window_start)
    errors=np.column_stack([data[f'q{i}_paper']-data[f'ref{i}_paper'] for i in range(1,8)])
    errors=(errors+np.pi)%(2*np.pi)-np.pi
    angle=data['q2_paper' if primitive=='S01' else 'q3_paper'][start:]
    active_time=time[start:]
    yaw=np.unwrap(data['yaw_rad'])
    if primitive=='S03':
        boundaries=[]
        for turn in range(int(np.floor(angle.min()/(2*np.pi)))+1,int(np.floor(angle.max()/(2*np.pi)))+1):
            hits=upward_crossings(active_time,angle,turn*2*np.pi)
            if hits: boundaries.append((hits[0],turn))
        complete=list(zip(boundaries[:-1],boundaries[1:]))
    else:
        center=.4 if primitive=='S01' else .63
        span=3.7 if primitive=='S01' else 1.22
        period=4 if primitive=='S01' else 2.44
        hits=upward_crossings(active_time,angle,center,refractory_s=period*.5)
        complete=[]
        for index,(left,right) in enumerate(zip(hits[:-1],hits[1:])):
            part=angle[(active_time>=left)&(active_time<=right)]
            if len(part) and part.min()<center-.25*span and part.max()>center+.25*span:
                complete.append(((left,index),(right,index+1)))
    cycles=[]
    for left,right in complete:
        a,b=left[0],right[0]
        # rotary只接受相邻2pi相位圈；不得把跳过的圈合称一圈。
        if primitive=='S03' and right[1]-left[1]!=1: continue
        delta=np.array([np.interp(b,time,xy[:,i])-np.interp(a,time,xy[:,i]) for i in range(2)])
        cycles.append(dict(cycle=len(cycles)+1,start_time_s=a,end_time_s=b,
                           forward_m=float(sign*delta[1]),lateral_m=float(delta[0]),
                           planar_net_m=float(np.linalg.norm(delta)),
                           start_yaw_rad=float(np.interp(a,time,yaw)),end_yaw_rad=float(np.interp(b,time,yaw))))
    metrics=dict(primitive=primitive,controller=manifest['controller'],trial=manifest['trial'],
                 total_steps=len(data),total_seconds=len(data)*.008,window_start_step=start,
                 actual_window_seconds=duration,forward_displacement_m=float(sign*displacement[1]),
                 lateral_displacement_m=float(displacement[0]),planar_displacement_m=float(np.linalg.norm(displacement)),
                 mean_forward_velocity_m_s=float(sign*displacement[1]/duration),
                 wrapped_reference_rmse_rad=float(np.sqrt(np.mean(errors[start:]**2))),
                 drive_min_rad=float(angle.min()),drive_max_rad=float(angle.max()),
                 drive_net_turns=float((angle[-1]-angle[0])/(2*np.pi)),
                 drive_negative_step_count=int((np.diff(angle)<-1e-4).sum()),
                 yaw_change_rad=float(yaw[-1]-yaw[start]),reset_events=manifest['reset_events'],
                 complete_measured_cycles=len(cycles),positive_complete_cycles=sum(c['forward_m']>0 for c in cycles),
                 all_states_finite=True,cycle_definition='actual upward phase crossings; partial intervals excluded',
                 statistical_unit='one paired deployment; cycles are correlated')
    (path/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    (path/'complete_measured_cycles.json').write_text(json.dumps(cycles,indent=2))
    if cycles:
        import csv
        with (path/'complete_measured_cycles.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(cycles[0]))
            writer.writeheader();writer.writerows(cycles)
    return metrics
