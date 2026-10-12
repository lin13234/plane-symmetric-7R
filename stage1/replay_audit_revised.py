import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.8 独立修订审计：重算36条原始回放，匹配40/41事件，审核双平面路径。

本程序复用 source_snapshot.py 的七维环面距离、原MDH/XML配准及720样本
驻留算法。快照保留原程序，不修改实验输入；新增的路径审计不改变分类阈值。
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import source_snapshot as core

OUT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parent))
PUBLICATION = Path('E:/02_mjlab/publication_2026_09')
DT = core.DT
COARSE = {**{f'P{x}_{s}': f'B_P_{s}' for x in ('0','pi') for s in ('pos','neg')},
          **{f'BS{k}': f'B_S{k}' for k in range(1,5)}}


def intervals(values, target):
    """返回目标状态真实连续区间，右端是样本覆盖区间的开端，不填充灰区。"""
    return [(int(a), int(b)) for a,b,v in core.runs(np.asarray(values)) if v == target]


def audit(recompute=True):
    """重分析全部真实数据，然后用事件身份而非总数之差做对应。"""
    core.OUT = OUT
    if recompute:
        core.analyse(ROOT, PUBLICATION)
    m = json.loads((OUT/'analysis_metrics.json').read_text(encoding='utf-8'))
    core.CFG = m['thresholds']
    sys.path.insert(0, str(ROOT))
    from planar_fourbar_dh.fourbar_dh import dh_parameters
    params = dh_parameters(1.)
    data = pd.read_csv(OUT/'branch_distances.csv')
    events = pd.read_csv(OUT/'branch_state_events.csv')
    fine = events[events.event_type.eq('state_change')].copy().reset_index(drop=True)
    fine['fine_event_id'] = [f'E8_{i+1:02d}' for i in range(len(fine))]
    old_path = ROOT/'stage1_replay_analysis/data/topology_v2/confirmed_state_changes_v2.csv'
    old = pd.read_csv(old_path)
    old['old_event_id'] = [f'E6_{i+1:02d}' for i in range(len(old))]
    original_audit = pd.read_csv(OUT/'interface_event_audit.csv')
    old_audit = pd.read_csv(OUT/'legacy_interface_audit_recomputed.csv')
    frames = {name: f.reset_index(drop=True) for name,f in data.groupby('replay',sort=False)}
    interfaces_df = pd.read_csv(ROOT/'outputs/gateway4_reflection_complete_rank_2026-09-17_183016/complete_rank_all_eight.csv')
    interfaces = {r.configuration: r[[f'q{i}_rad' for i in range(1,8)]].to_numpy(float) for _,r in interfaces_df.iterrows()}

    # 一一匹配每条回放的粗事件序列。P0/Pπ合并后自变化不参与匹配；
    # 必须同时核对前后标签和序列次序，避免仅根据40与41计数推断来源。
    correspondence = []
    matched_old = set()
    for name, group in fine.groupby('replay',sort=False):
        old_group = old[old.replay.eq(name)].reset_index(drop=True)
        cursor = 0
        for _, e in group.iterrows():
            a,b = COARSE[e.from_branch],COARSE[e.to_branch]
            r = e.to_dict()
            r.update(coarse_from=a, coarse_to=b, involves_planar_split=a==b,
                     matching_rule='replay + ordered coarse from/to identity')
            if a != b:
                if cursor >= len(old_group):
                    raise AssertionError(f'{name}: extra inter-coarse event')
                o = old_group.iloc[cursor]; cursor += 1
                if (a,b) != (o.from_branch,o.to_branch):
                    raise AssertionError(f'{name}: coarse identity/order mismatch')
                matched_old.add(o.old_event_id)
                r.update(old_event_id=o.old_event_id,old_confirmation_frame=int(o.confirmation_frame),
                         old_confirmation_time_s=float(o.confirmation_time),
                         confirmation_shift_samples=int(e.confirmation_frame-o.confirmation_frame),
                         correspondence='matched original coarse event')
                oa=old_audit[(old_audit.replay==name)&(old_audit.confirmation_frame==o.confirmation_frame)].iloc[0]
                r['old_topology_candidate']=bool(oa.topology_consistent_candidate)
                r['old_strong_rank_pass']=bool(oa.legacy_absolute_rank_test)
            else:
                r.update(old_event_id='',old_confirmation_frame=np.nan,old_confirmation_time_s=np.nan,
                         confirmation_shift_samples=np.nan,correspondence='extra internal planar-label change')
            f=frames[name]; i=int(e.confirmation_frame); start=int(e.support_start_frame)
            for prefix,j in [('support_start',start),('confirmation',i)]:
                for branch in (e.from_branch,e.to_branch):
                    r[f'{prefix}_d_{branch}_rad']=float(f.loc[j,'d_'+branch])
            ar=original_audit[(original_audit.replay==name)&(original_audit.confirmation_frame==i)].iloc[0]
            for c in ['nearest_expected_interface','interface_distance_rad','candidate_time_s',
                      'rank_Jc_relative_1e_10','topology_consistent_candidate','strict_rank_five_verified',
                      'closure_valid_window','max_q_step_rad']:
                r[c]=ar[c]
            # 无直接关联接口时不能沿用旧函数的窗口起点作为“接口雅可比”。
            if a==b:
                for c in ['nearest_expected_interface','interface_distance_rad','candidate_time_s','rank_Jc_relative_1e_10']:
                    r[c]=np.nan
                r['audit_note']='No direct planar-planar interface; see spatial-route audit below.'
            correspondence.append(r)
        if cursor != len(old_group):
            raise AssertionError(f'{name}: unmatched old event')
    if matched_old != set(old.old_event_id):
        raise AssertionError('Not all 40 old events have been individually matched')

    # 逐种子、逐平面循环保存确认事件与当前证据支持区间；held标签不算持续占据。
    planar_rows=[]
    for name,f in frames.items():
        if f.condition.iloc[0] != 'A1': continue
        side=f.assembly.iloc[0]
        seq=events[events.replay.eq(name)]
        q=f[[f'q{i}_rad' for i in range(1,8)]].to_numpy()
        for branch in ('P0_'+side,'Ppi_'+side):
            locks=seq[seq.to_branch.eq(branch)]
            support=f.raw_label.eq(branch)&f.confirmed_branch.eq(branch)
            episodes=intervals(np.where(support,branch,'UNSUPPORTED'),branch)
            raw_episodes=intervals(f.raw_label.to_numpy(),branch)
            ids=np.flatnonzero(support)
            row={'record_type':'planar occupancy','replay':name,'assembly':side,'condition':'A1',
                 'seed':int(f.seed.iloc[0]),'planar_cycle':branch,
                 'confirmed_occupancy':bool(len(locks)),
                 'classification':'Confirmed planar-cycle occupancy' if len(locks) else 'Unconfirmed / ambiguous',
                 'confirmation_times_s':json.dumps(locks.confirmation_time_s.tolist()),
                 'dwell_evidence_start_times_s':json.dumps(locks.support_start_time_s.tolist()),
                 'current_supported_intervals_s':json.dumps([[a*DT,b*DT] for a,b in episodes]),
                 'current_supported_duration_s':len(ids)*DT,
                 'maximum_raw_contiguous_duration_s':max([b-a for a,b in raw_episodes],default=0)*DT,
                 'complete_closure_within_frozen_classifier_tolerance':bool(((f.closure_position_m<=core.CFG['tau_closure_pos'])&(f.closure_orientation_chordal<=core.CFG['tau_closure_ang'])).all()),
                 'max_adjacent_joint_increment_rad':float(np.linalg.norm(core.wrap(np.diff(q,axis=0)),axis=1).max()),
                 'confirmed_sequence':' -> '.join(seq.to_branch),
                 'verified_planar_to_planar_traversal':False}
            if len(ids):
                row.update(min_supported_branch_distance_rad=float(f.loc[ids,'d_'+branch].min()),
                           max_supported_branch_distance_rad=float(f.loc[ids,'d_'+branch].max()))
            planar_rows.append(row)

    # 对新增平面标签变化审核两个理论路径：Pπ -> Hk -> BSk -> Gk -> P0。
    # 窗口从最后一段旧平面证据的末720样本开始，至新平面确认结束，
    # 覆盖实际桥接过程，不能只检视新标签前一个驻留窗口。
    routes=[]; extra_trace=None
    for r in correspondence:
        if not r['involves_planar_split']: continue
        f=frames[r['replay']]; end=int(r['confirmation_frame'])+1
        before=np.flatnonzero(f.raw_label.iloc[:int(r['support_start_frame'])].eq(r['from_branch']))
        last=int(before[-1]); left=max(0,last-core.CFG['n_dwell']+1)
        q=f[[f'q{i}_rad' for i in range(1,8)]].to_numpy()
        w=f.iloc[left:end].copy()
        _,jc,_=core.closure_jacobians(q[left:end],params)
        sv=np.linalg.svd(jc,compute_uv=False)
        w['sigma6_closure_Jc']=sv[:,-1];w['sigma1_closure_Jc']=sv[:,0]
        w['sigma6_over_sigma1']=sv[:,-1]/sv[:,0]
        w['rank_Jc_relative_1e_10']=(sv/sv[:,0,None]>1e-10).sum(axis=1)
        extra_trace=w
        indices=(1,2) if r['assembly']=='pos' else (3,4)
        for k in indices:
            entering,leaving=(f'H{k}',f'G{k}') if r['from_branch'].startswith('Ppi') else (f'G{k}',f'H{k}')
            ie=int(w['d_'+entering].idxmin()); il=int(w['d_'+leaving].idxmin())
            raw_blocks=intervals(w.raw_label.to_numpy(),f'BS{k}')
            longest=max([b-a for a,b in raw_blocks],default=0)
            confirmed_spatial=bool((w.confirmed_branch.eq(f'BS{k}')&w.raw_label.eq(f'BS{k}')).any())
            close=bool(w['d_'+entering].min()<core.CFG['tau_gateway'] and w['d_'+leaving].min()<core.CFG['tau_gateway'])
            order=ie<il
            rank5both=bool(w.loc[ie,'rank_Jc_relative_1e_10']==5 and w.loc[il,'rank_Jc_relative_1e_10']==5)
            # 两端距离、顺序、空间驻留均满足才是该完整路径的候选；本数据均未通过。
            route_candidate=bool(close and order and confirmed_spatial)
            row={'record_type':'sequential planar route audit','replay':r['replay'],'assembly':r['assembly'],
                 'condition':r['condition'],'seed':r['seed'],'from_branch':r['from_branch'],'to_branch':r['to_branch'],
                 'route':f'{r["from_branch"]} -> {entering} -> BS{k} -> {leaving} -> {r["to_branch"]}',
                 'window_start_s':left*DT,'window_end_s_exclusive':end*DT,
                 'last_from_planar_raw_time_s':last*DT,'new_planar_support_start_s':r['support_start_time_s'],
                 'new_planar_confirmation_s':r['confirmation_time_s'],
                 'entering_interface':entering,'leaving_interface':leaving,
                 'entering_interface_time_s':ie*DT,'leaving_interface_time_s':il*DT,
                 'entering_interface_distance_rad':float(w.loc[ie,'d_'+entering]),
                 'leaving_interface_distance_rad':float(w.loc[il,'d_'+leaving]),
                 'entering_rank_Jc':int(w.loc[ie,'rank_Jc_relative_1e_10']),
                 'leaving_rank_Jc':int(w.loc[il,'rank_Jc_relative_1e_10']),
                 'entering_sigma6':float(w.loc[ie,'sigma6_closure_Jc']),
                 'leaving_sigma6':float(w.loc[il,'sigma6_closure_Jc']),
                 'max_contiguous_raw_spatial_samples':longest,'max_contiguous_raw_spatial_duration_s':longest*DT,
                 'total_raw_spatial_samples':int(w.raw_label.eq(f'BS{k}').sum()),
                 'confirmed_spatial_between_planar_states':confirmed_spatial,
                 'interface_order_correct':order,'both_interfaces_within_0p15_rad':close,
                 'complete_closure_within_frozen_classifier_tolerance':bool(((w.closure_position_m<=core.CFG['tau_closure_pos'])&(w.closure_orientation_chordal<=core.CFG['tau_closure_ang'])).all()),
                 'max_adjacent_joint_increment_rad':float(w.q_step_rad.iloc[1:].max()),
                 'minimum_Jc_rank_in_window':int(w.rank_Jc_relative_1e_10.min()),
                 'both_rank_five':rank5both,'topology_consistent_transition_candidate':route_candidate,
                 'verified_planar_to_planar_traversal':False,
                 'classification':'Sequential planar-cycle labels; continuous traversal not verified'}
            routes.append(row);planar_rows.append(row)
        r.update(audit_window_start_s=left*DT,audit_window_end_s_exclusive=end*DT,
                 bridge_raw_BS3_samples=int(w.raw_label.eq('BS3').sum()),
                 bridge_max_raw_BS3_samples=max([b-a for a,b in intervals(w.raw_label.to_numpy(),'BS3')],default=0),
                 bridge_G3_min_rad=float(w.d_G3.min()),bridge_H3_min_rad=float(w.d_H3.min()),
                 bridge_G3_rank=int(w.loc[w.d_G3.idxmin(),'rank_Jc_relative_1e_10']),
                 bridge_H3_rank=int(w.loc[w.d_H3.idxmin(),'rank_Jc_relative_1e_10']),
                 final_verdict='Sequential planar labels; coarse split; no verified continuous traversal')
    pd.DataFrame(correspondence).to_csv(OUT/'six_vs_eight_labels.csv',index=False)
    pd.DataFrame(planar_rows).to_csv(OUT/'planar_cycle_audit.csv',index=False)
    pd.DataFrame(routes).to_csv(OUT/'planar_route_audit.csv',index=False)
    if extra_trace is not None: extra_trace.to_csv(OUT/'additional_planar_event_trace.csv',index=False)

    # 主图始终保留原有G3、A1、seed2026案例；接口距离固定为相关G3。
    chosen=original_audit[original_audit.topology_consistent_candidate]
    if len(chosen)!=1 or chosen.iloc[0].replay!='G3_A1_seed2026':
        raise AssertionError('Representative original candidate changed; must report rather than silently replace')
    e=chosen.iloc[0]; f=frames[e.replay]; a,b=int(e.window_start_frame),int(e.window_end_frame_exclusive)
    q=f[[f'q{i}_rad' for i in range(1,8)]].to_numpy()
    _,jc,screw=core.closure_jacobians(q[a:b],params)
    sv=np.linalg.svd(jc,compute_uv=False); ss=np.linalg.svd(screw,compute_uv=False)
    trace=f.iloc[a:b][['replay','time_s','closure_position_m','closure_orientation_rad',
                           'closure_orientation_chordal','closure_MDH_position_over_l','q_step_rad','d_G3']].copy()
    trace['distance_relevant_G3_rad']=trace.d_G3
    trace['sigma6_closure_Jc']=sv[:,-1];trace['sigma1_closure_Jc']=sv[:,0]
    trace['sigma6_over_sigma1']=sv[:,-1]/sv[:,0]
    trace['sigma6_screw_J']=ss[:,-1]
    trace['rank_Jc_relative_1e_10']=(sv/sv[:,0,None]>1e-10).sum(axis=1)
    trace['candidate_marker']=np.isclose(trace.time_s,float(e.candidate_time_s),atol=1e-10)
    trace.to_csv(OUT/'interface_audit.csv',index=False)
    # 明确灰色的不同成因，但不改变原分类与滞回逻辑。
    support=data.raw_label.eq(data.confirmed_branch)
    gray_reason=np.where(data.raw_label.eq('INTERFACE_AMBIGUOUS'),'interface ambiguity',
                np.where(data.raw_label.eq('INVALID'),'closure rejection',
                np.where(data.raw_label.isin(core.NAMES),'dwell pending / historical mismatch',
                np.where(data.nearest_distance_rad>core.CFG['tau_d'],'distance threshold exceeded','separation margin insufficient'))))
    data['timeline_label']=np.where(support,data.confirmed_branch,'UNSUPPORTED')
    data['gray_reason']=np.where(support,'supported confirmed state',gray_reason)
    data.to_csv(OUT/'audited_replays.csv.gz',index=False,compression='gzip',float_format='%.12g')
    gray=data[~support].groupby(['replay','gray_reason']).size().reset_index(name='samples')
    gray['duration_s']=gray.samples*DT;gray.to_csv(OUT/'gray_interval_summary.csv',index=False)
    fine.to_csv(OUT/'eight_cycle_changes.csv',index=False)
    # 所有41事件的修订版审计：无直接接口的额外事件不显示伪“候选点”。
    no_interface=original_audit.nearest_expected_interface.isna()|original_audit.nearest_expected_interface.eq('')
    original_audit.loc[no_interface,['candidate_frame','candidate_time_s','sigma6_closure_Jc',
                                    'sigma6_screw_J','rank_Jc_relative_1e_10']]=np.nan
    original_audit['verification_statement']=np.where(original_audit.topology_consistent_candidate,
        'Rank-6 candidate; not verified','Dwell-confirmed label change; physical switch not verified')
    original_audit.to_csv(OUT/'interface_event_audit.csv',index=False)
    mapping=pd.DataFrame(correspondence)
    validation={'old_events_individually_matched':len(matched_old),'fine_state_changes':len(fine),
                'additional_events':len(mapping[mapping.involves_planar_split]),
                'old_confirmation_shift_min_samples':int(mapping.confirmation_shift_samples.min()),
                'old_confirmation_shift_max_samples':int(mapping.confirmation_shift_samples.max()),
                'topology_candidates':int(original_audit.topology_consistent_candidate.sum()),
                'strictly_verified_switches':int(original_audit.strict_rank_five_verified.sum()),
                'planar_routes':routes,'main_candidate':e.to_dict(),
                'main_candidate_window_rank_min':int(trace.rank_Jc_relative_1e_10.min()),
                'main_candidate_window_rank_max':int(trace.rank_Jc_relative_1e_10.max()),
                'a1_distinct_confirmed_planar_counts':{branch:int(sum(v.get('planar_cycle')==branch and v.get('confirmed_occupancy',False) for v in planar_rows)) for branch in core.NAMES[:4]}}
    (OUT/'revision_metrics.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
    print(json.dumps({k:v for k,v in validation.items() if k not in ('planar_routes','main_candidate')},indent=2),flush=True)
    return validation


if __name__=='__main__':
    audit('--reuse-recomputed' not in sys.argv)
