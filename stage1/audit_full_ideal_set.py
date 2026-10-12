import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A3 灰区的独立完整解集审计；只读取真实 qpos，不运行训练或模拟。

输出 science_closure_review/：全部12循环距离、两种闭环切口残差、
灰区原因、H侧事件审计及数值交叉验证。保留冻结八循环分类与720驻留。
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar
import source_snapshot as core

BASE = Path(__file__).resolve().parent
OUT = BASE/'science_closure_review'
ROOT = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parent))
PUB = Path('E:/02_mjlab/publication_2026_09')
L_NORM = .060  # 米：归一化尺度与本批次Stage-I XML实际轴间距相符。
DT = core.DT


def gray_codes(f):
    """灰区区分：距离越界、模型检查失败、接口/分离歧义、驻留待确认。

    'outside ideal tube' 仅指超过冻结距离阈值；不声称找到新理想分支。
    冻结闭环容差是宽松分类门限，不能作为精确理想闭环的证明。
    """
    supported = f.raw_label.eq(f.confirmed_branch).to_numpy()
    d12 = np.minimum(f.nearest_distance_rad, f.nearest_folded_distance_rad)
    result = np.full(len(f), 'pending dwell', dtype=object)
    result[f.raw_label.isin(['INTERFACE_AMBIGUOUS','AMBIGUOUS']).to_numpy()] = 'ambiguous'
    result[(f.raw_label.eq('NONE') & (f.margin_rad < .015)).to_numpy()] = 'ambiguous'
    result[np.asarray(d12 > .055)] = 'outside ideal tube'
    result[f.raw_label.eq('INVALID').to_numpy()] = 'closure gate failed'
    result[supported] = 'supported'
    return result


def main():
    OUT.mkdir(exist_ok=True)
    sys.path[:0] = [str(ROOT),str(PUB/'00_joint_registration'),str(PUB/'02_branch_evaluator')]
    from planar_fourbar_dh.fourbar_dh import solve_explicit, dh_parameters
    from calibrate_registration import XmlChainFK, XML_PATH
    from registration import xml_to_paper_q, paper_to_xml_qpos
    params = dh_parameters(1.)
    fk = XmlChainFK(XML_PATH)
    inputs = {}
    def remember(p,role):
        p=Path(p).resolve()
        inputs[str(p)]={'role':role,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        return p
    for p,role in [(XML_PATH,'frozen Stage-I XML'),
        (ROOT/'planar_fourbar_dh/fourbar_dh.py','authoritative MDH and planar solver'),
        (PUB/'02_branch_evaluator/registration.py','joint registration'),
        (PUB/'00_joint_registration/calibrate_registration.py','independent XML FK'),
        (ROOT/'Section3_Topology_Proof_20261004/supplement_full7R_topology_and_coverage.tex','folded-circle formula'),
        (BASE/'source_snapshot.py','retained analytical distance implementation'),
        (Path(__file__),'independent complete-set diagnostic implementation'),
        (ROOT/'stage1_replay_analysis/data/summary/classifier_config.json','frozen classification criteria'),
        (BASE/'six_vs_eight_labels.csv','event-level correspondence'),
        (BASE/'branch_state_events.csv','frozen dwell events')]:
        remember(p,role)
    s=np.linspace(0,2*np.pi,721)
    branches={}
    for sign,side in [(1,'pos'),(-1,'neg')]:
        base=solve_explicit(s,branch=sign).q[:,[6,0,1,2,3,4,5]]
        branches['P0_'+side]=core.Branch('P0_'+side,s,base)
        h=np.array([0,0,np.pi,np.pi,0,np.pi,np.pi])+base*np.array([1,1,1,-1,1,-1,1])
        branches['Ppi_'+side]=core.Branch('Ppi_'+side,s,h)
    for k in range(1,5):
        p=remember(PUB/f'01_branch_tables/branch_table_B_S{k}.csv',f'BS{k} analytical continuation')
        df=pd.read_csv(p)
        branches[f'BS{k}']=core.Branch(f'BS{k}',df.s_rad,df[[f'q{i}_rad' for i in range(1,8)]].to_numpy())
    for a in (0,1):
        for eps in (-1,1):
            th=np.arctan2((-5-8*eps*np.sqrt(10))/41,(-4+10*eps*np.sqrt(10))/41)
            q=np.tile([th,th,a*np.pi,0,np.pi,0,(a+1)*np.pi],(len(s),1))
            q[:,3]=s; q[:,5]=(-1)**a*2*th-s
            n=f'F{a}_{eps}'
            branches[n]=core.Branch(n,s,q)
    # 每条参考循环的节点及中点用MDH和XML分别验证；检查对象不是RL插值。
    refs=[]
    for name,b in branches.items():
        qq=b.evaluate(np.r_[b.s,(b.s[:-1]+b.s[1:])/2])
        T,_,_=core.mdh_batch(qq,params)
        pp,ch=fk.closure(paper_to_xml_qpos(qq))
        pos=np.linalg.norm(T[:,:3,3],axis=1)
        ang=np.linalg.norm(Rotation.from_matrix(T[:,:3,:3]).as_rotvec(),axis=1)
        assert pos.max()<1e-6 and ang.max()<1e-6
        refs.append({'branch':name,'samples':len(qq),'MDH_position_over_Lnorm_max':pos.max(),
            'orientation_rad_max':ang.max(),'XML_position_m_max':pp.max(),'XML_chordal_max':ch.max()})
    pd.DataFrame(refs).to_csv(OUT/'ideal_12_reference_validation.csv',index=False)
    reference_rows=[]
    for n,b in branches.items():
        for phase,qi in zip(b.s,b.q):
            reference_rows.append({'branch':n,'s_rad':phase,**{f'q{i+1}_rad':qi[i] for i in range(7)}})
    pd.DataFrame(reference_rows).to_csv(OUT/'ideal_12_reference_samples.csv',index=False)
    interfacepath=remember(ROOT/'outputs/gateway4_reflection_complete_rank_2026-09-17_183016/complete_rank_all_eight.csv','eight complete G/H configurations')
    iface=pd.read_csv(interfacepath)
    interfaces={r.configuration:r[[f'q{i}_rad' for i in range(1,8)]].to_numpy(float) for _,r in iface.iterrows()}
    # 已审计数据仅用于交叉核对，不作为重算q或距离的替代品。
    cached=pd.read_csv(BASE/'audited_replays.csv.gz')
    events=pd.read_csv(BASE/'branch_state_events.csv')
    allframes=[]; summaries=[]; episodes=[]; hrows=[]; numeric=[]; cut_checks=[]
    for side,tag,folder in [('pos','G1','runs'),('neg','G3','runs_g3init')]:
        for seed in (7,42,2026):
            name=f'{tag}_A3_seed{seed}'
            rawpath=remember(PUB/'05_pilots'/folder/f'Stage1_A3_{"G3Init_" if side=="neg" else ""}seed{seed}'/'replay/replay_qpos.csv','raw seven-joint replay')
            qxml=pd.read_csv(rawpath)[[f'qpos{i}' for i in range(1,8)]].to_numpy(float)[:-1]
            q=xml_to_paper_q(qxml)
            f=cached[cached.replay.eq(name)].reset_index(drop=True).copy()
            assert q.shape==(9999,7)
            assert np.max(np.abs(core.wrap(q-f[[f'q{i}_rad' for i in range(1,8)]].to_numpy())))<1e-10
            # 新诊断表保留原始qpos映射的完整浮点值，旧缓存仅用作交叉核对。
            for i in range(7):f[f'q{i+1}_rad']=q[:,i]
            d=[]
            for n,b in branches.items():
                dd,grid,phase=b.distance(q)
                f['d_'+n]=dd; d.append(dd)
                # 抽查极端灰区、最大距离及均匀时刻：穷举表格+独立有界优化。
                ids=np.unique(np.r_[np.linspace(0,len(q)-1,9,dtype=int),np.argmax(dd)])
                brute=np.linalg.norm(core.wrap(q[ids,None]-b.q[None]),axis=2).min(axis=1)
                err=np.max(np.abs(brute-grid[ids])); assert err<1e-10
                independent=[]
                for ii in ids:
                    start=b.s[int(b.tree.query(np.mod(q[ii],2*np.pi))[1])]
                    opt=minimize_scalar(lambda u:np.linalg.norm(core.wrap(q[ii]-b.evaluate(u))),
                        bounds=(start-b.step,start+b.step),method='bounded',options={'xatol':1e-13})
                    independent.append(abs(min(float(opt.fun),float(grid[ii]))-dd[ii]))
                numeric.append({'replay':name,'branch':n,'grid_bruteforce_error_rad':err,
                    'independent_local_max_difference_rad':max(independent)})
            dm=np.column_stack(d); f['d_min_12_rad']=dm.min(axis=1)
            f['nearest_ideal_cycle']=np.array(list(branches))[dm.argmin(axis=1)]
            f['nearest_distance_rad']=dm[:,:8].min(axis=1)
            f['nearest_folded_distance_rad']=dm[:,8:].min(axis=1)
            pg,ch=fk.closure(qxml)
            T,J,_=core.closure_jacobians(q,params)
            # 独立核验两个闭环切口的固定坐标变换，不能把不同原点的位置残差混用。
            # XML关节3高度0.0175m使MDH闭环原点相对weld-site原点偏移c。
            P,R=fk.fk(qxml)
            ba,oa=fk._find_site('site_A');bb,ob=fk._find_site('site_dummy')
            wa=P[ba]+R[ba]@oa;wb=P[bb]+R[bb]@ob
            RaT=R[ba].transpose(0,2,1)
            px=np.einsum('nij,nj->ni',RaT,wb-wa)
            Rx=RaT@R[bb]
            c=np.array([0.,0.,-fk.bodies['L-link2'].pos[2]])
            assert abs(c[2]+.0175)<1e-14
            predicted=px+(np.eye(3)[None]-Rx)@c
            pos_error=np.max(np.linalg.norm(L_NORM*T[:,:3,3]-predicted,axis=1))
            rot_error=np.max(np.abs(T[:,:3,:3]-Rx))
            assert pos_error<1e-12 and rot_error<1e-12
            cut_checks.append({'replay':name,'c_x_m':c[0],'c_y_m':c[1],'c_z_m':c[2],
                'max_position_transform_error_m':pos_error,'max_rotation_matrix_error':rot_error,
                'relation':'p_MDH = p_XML + (I - R_XML) c; R_MDH = R_XML'})
            theta=np.linalg.norm(Rotation.from_matrix(T[:,:3,:3]).as_rotvec(),axis=1)
            sv=np.linalg.svd(J,compute_uv=False)
            f['closure_XML_position_m']=pg
            f['closure_XML_position_over_Lnorm']=pg/L_NORM
            f['closure_MDH_position_m']=L_NORM*np.linalg.norm(T[:,:3,3],axis=1)
            f['closure_MDH_position_over_Lnorm']=np.linalg.norm(T[:,:3,3],axis=1)
            f['closure_orientation_rad']=theta
            f['closure_orientation_chordal']=ch
            f['sigma6_over_sigma1']=sv[:,-1]/sv[:,0]
            f['rank_Jc_relative_1e_10']=(sv/sv[:,0,None]>1e-10).sum(axis=1)
            f['gray_scientific_reason']=gray_codes(f)
            assert np.max(np.abs(theta-2*np.arcsin(np.clip(ch/(2*np.sqrt(2)),0,1))))<1e-10
            counts=f.gray_scientific_reason.value_counts().to_dict()
            row={'replay':name,'assembly':side,'seed':seed,'samples':len(f),
                'd_min_12_max_rad':f.d_min_12_rad.max(),'nearest_folded_min_rad':f.nearest_folded_distance_rad.min(),
                'XML_position_max_m':pg.max(),'MDH_position_max_m':f.closure_MDH_position_m.max(),
                'MDH_position_over_Lnorm_max':f.closure_MDH_position_over_Lnorm.max(),
                'orientation_max_rad':theta.max(),'q_step_max_rad':f.q_step_rad.max(),
                'rank5_samples':int((f.rank_Jc_relative_1e_10==5).sum()),
                'dwell_sequence':' -> '.join(events[events.replay.eq(name)].to_branch)}
            for reason in ['supported','pending dwell','ambiguous','outside ideal tube','closure gate failed']:
                row[reason.replace(' ','_')+'_samples']=counts.get(reason,0)
                row[reason.replace(' ','_')+'_duration_s']=counts.get(reason,0)*DT
            summaries.append(row)
            for a,b,reason in core.runs(f.gray_scientific_reason.to_numpy()):
                if reason=='supported':continue
                w=f.iloc[a:b]
                episodes.append({'replay':name,'seed':seed,'start_s':a*DT,'end_s_exclusive':b*DT,
                    'samples':b-a,'duration_s':(b-a)*DT,'reason':reason,
                    'd_min_12_min_rad':w.d_min_12_rad.min(),'d_min_12_max_rad':w.d_min_12_rad.max(),
                    'XML_position_max_m':w.closure_XML_position_m.max(),
                    'MDH_position_max_m':w.closure_MDH_position_m.max(),
                    'orientation_max_rad':w.closure_orientation_rad.max()})
            for _,e in events[events.replay.eq(name)&events.event_type.eq('state_change')].iterrows():
                k=int(e.from_branch[-1]) if e.from_branch.startswith('BS') else int(e.to_branch[-1])
                expected=f'H{k}'
                # 扩展桥接诊断：从旧状态最后支持段前720样本到新状态确认。
                # 冻结候选统计仍使用原窗口/判据，不以扩窗暗中改变1/0统计。
                last=np.flatnonzero(f.raw_label.iloc[:int(e.support_start_frame)].eq(e.from_branch))[-1]
                left=max(0,int(last)-719);right=int(e.confirmation_frame)+1
                w=f.iloc[left:right]
                dh=np.linalg.norm(core.wrap(q[left:right]-interfaces[expected]),axis=1)
                j=left+int(np.argmin(dh))
                hr={'replay':name,'assembly':side,'seed':seed,'from_branch':e.from_branch,'to_branch':e.to_branch,
                    'confirmation_time_s':e.confirmation_time_s,'support_start_time_s':e.support_start_time_s,
                    'window_start_s':left*DT,'window_end_s_exclusive':right*DT,'expected_interface':expected,
                    'nearest_H_time_s':j*DT,'nearest_H_distance_rad':float(dh.min()),
                    'near_H_within_0p15_rad':bool(dh.min()<.15),
                    'rank_at_nearest_H':int(f.rank_Jc_relative_1e_10.iloc[j]),
                    'sigma6_over_sigma1_at_nearest_H':float(f.sigma6_over_sigma1.iloc[j]),
                    'minimum_sigma6_over_sigma1_window':float(w.sigma6_over_sigma1.min()),
                    'd_min_12_at_nearest_H_rad':float(f.d_min_12_rad.iloc[j]),
                    'XML_position_at_nearest_H_m':float(pg[j]),'MDH_position_at_nearest_H_m':float(f.closure_MDH_position_m.iloc[j]),
                    'orientation_at_nearest_H_rad':float(theta[j]),
                    'outside_ideal_tube_samples':int(w.gray_scientific_reason.eq('outside ideal tube').sum()),
                    'rank5_samples_window':int(w.rank_Jc_relative_1e_10.eq(5).sum()),
                    'max_q_step_window_rad':float(w.q_step_rad.max()),'verified_continuous_switch':False,
                    'verdict':'Sequentially detected branch states; no verified rank-five traversal'}
                hrows.append(hr)
            allframes.append(f)
            print(name,'d12max',row['d_min_12_max_rad'],'gray',counts,flush=True)
    pd.concat(allframes,ignore_index=True).to_csv(OUT/'A3_all12_distances_and_closure.csv.gz',index=False,compression='gzip')
    pd.DataFrame(summaries).to_csv(OUT/'A3_gray_diagnostic_summary.csv',index=False)
    pd.DataFrame(episodes).to_csv(OUT/'A3_gray_intervals.csv',index=False)
    pd.DataFrame(hrows).to_csv(OUT/'A3_H_interface_audit.csv',index=False)
    pd.DataFrame(numeric).to_csv(OUT/'distance_cross_validation.csv',index=False)
    pd.DataFrame(cut_checks).to_csv(OUT/'closure_cut_transform_validation.csv',index=False)
    # 再次核对事件身份；避免凭40/41总差把新增事件错误归因。
    mapping=pd.read_csv(BASE/'six_vs_eight_labels.csv')
    old=pd.read_csv(remember(ROOT/'stage1_replay_analysis/data/topology_v2/confirmed_state_changes_v2.csv','all 40 original coarse events'))
    matched=mapping[mapping.old_event_id.notna()]
    assert len(old)==len(matched)==40 and matched.old_event_id.nunique()==40
    for _,r in matched.iterrows():
        o=old[old.replay.eq(r.replay)&old.confirmation_frame.eq(r.old_confirmation_frame)]
        assert len(o)==1 and o.iloc[0].from_branch==r.coarse_from and o.iloc[0].to_branch==r.coarse_to
    extra=mapping[mapping.old_event_id.isna()]
    assert len(extra)==1 and extra.iloc[0].replay=='G3_A1_seed7'
    metrics={'L_norm_m':L_NORM,'n_A3_replays':6,'samples_per_replay':9999,'ideal_cycles':list(branches),
        'tau_d_rad':.055,'tau_interface_candidate_rad':.15,'rank_relative_tolerance':1e-10,
        'all_40_original_event_identities_rechecked':True,'extra_event':extra.iloc[0].to_dict(),
        'reference_max_position_over_Lnorm':max(r['MDH_position_over_Lnorm_max'] for r in refs),
        'reference_max_orientation_rad':max(r['orientation_rad_max'] for r in refs),
        'all_A3_rank5_samples':sum(r['rank5_samples'] for r in summaries),
        'max_independent_local_difference_rad':max(r['independent_local_max_difference_rad'] for r in numeric),
        'max_closure_cut_position_transform_error_m':max(r['max_position_transform_error_m'] for r in cut_checks),
        'max_closure_cut_rotation_matrix_error':max(r['max_rotation_matrix_error'] for r in cut_checks),
        'no_RL_interpolation':True,'no_threshold_changes':True,'summaries':summaries,'H_events':hrows}
    (OUT/'diagnostic_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (OUT/'input_manifest.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')


if __name__=='__main__':main()
