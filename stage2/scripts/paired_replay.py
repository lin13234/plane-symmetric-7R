"""一条固定配对条件的确定性部署；保存完整状态，失败不重试。

预检只运行160步，不产生正式性能结论。正式部署严格使用原语的
42/64秒预算，没有视频渲染开销，也不拼接自动reset后的路线。
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
import mujoco
from protocol_support import (ROOT,TRAIN,XMLS,sha,check_sources,paired_config,
                              checkpoint_path,actor_mean,device_identity,paper_angles)

FIELDS=('qpos','qvel','qacc','qacc_warmstart','ctrl','qfrc_actuator','qfrc_constraint','nefc')


def capture(raw):
    """同步复制数组，不能以广义约束力替代逐接触力。"""
    result={name:getattr(raw.sim.data,name)[0].detach().cpu().numpy().copy() for name in FIELDS}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('状态/求解器记录出现非有限数值')
    return result


def nominal_paper(primitive,term):
    """使用实际动作应用时钟，不能以另一个时钟生成误差参考。"""
    if primitive=='S01':
        from mjlab.tasks.stage2_joint2_rebuilt.reference import joint2_reference_torch
        xml=joint2_reference_torch(term._phase)[0].detach().cpu().numpy()
        result=xml[[6,0,1,2,3,4,5]].copy()
        result[[3,6]]*=-1
        return float(term._phase[0].item()),result
    phase=term.phase
    value,_=term.reference.paper_targets(phase,term.phase_velocity)
    return float(phase[0].item()),value[0].detach().cpu().numpy().copy()


def record(args):
    """初态读回、物理配对和有限性均成立后才允许开始正式采集。"""
    check_sources()
    if args.out.exists(): raise FileExistsError('禁止覆盖已有记录：'+str(args.out))
    args.out.mkdir(parents=True)
    cfg,entity,requested_q,requested_v,perturbation=paired_config(args.primitive,args.trial)
    identity=device_identity()
    checkpoint=checkpoint_path(args.primitive,args.controller)
    actor=actor_mean(checkpoint,'cuda:0')
    steps=160 if args.precheck else (5250 if args.primitive=='S02' else 8000)
    manifest=dict(status='initializing',scope='precheck' if args.precheck else 'formal_paired_deployment',
                  primitive=args.primitive,controller=args.controller,trial=args.trial,
                  gpu=identity,requested_steps=steps,completed_steps=0,step_dt=.008,
                  checkpoint=None if checkpoint is None else str(checkpoint),
                  checkpoint_sha256=None if checkpoint is None else sha(checkpoint),
                  weights_loaded=checkpoint is not None,xml_sha256=sha(XMLS[args.primitive]),
                  perturbation=perturbation,world_forward_axis=[0,-1 if args.primitive=='S01' else 1],
                  residual_scale_rad=.02 if args.primitive=='S01' else .05,
                  distance_safety_disabled=True,workspace=dict(nconmax=256,njmax=1024,ls_parallel=False),
                  automatic_retry=False,video_recorded=False,reset_events=[])
    path=args.out/'manifest.json'
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    trace={name:[] for name in FIELDS}
    rows,actions,residuals=[],[],[]
    raw=None
    initial={}
    error=None
    started=time.monotonic()
    try:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.stage2_skill.command import SkillCommand
        torch.manual_seed(42)
        np.random.seed(42)
        raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        assert abs(raw.step_dt-.008)<1e-12
        vec=RslRlVecEnvWrapper(raw,clip_actions=None)
        command=None
        if args.primitive!='S01':
            command=raw.command_manager.get_term('skill')
            command.set_command(torch.arange(1,device=raw.device),SkillCommand(command.cfg.skill_id),reset_phase=True)
        initial=capture(raw)
        initial['geom_friction']=raw.sim.mj_model.geom_friction.copy()
        np.savez_compressed(args.out/'pre_step_initial_state.npz',**initial)
        qerror=float(np.max(np.abs(initial['qpos']-requested_q)))
        verror=float(np.max(np.abs(initial['qvel']-requested_v)))
        assert qerror<2e-6 and verror<2e-6, 'GPU实际reset不同于配对初态'
        baseline=mujoco.MjModel.from_xml_path(str(XMLS[args.primitive]))
        friction=baseline.geom_friction.copy()
        collidable=(baseline.geom_contype!=0)|(baseline.geom_conaffinity!=0)
        friction[collidable,0]*=float(perturbation['sliding_friction_multiplier'])
        assert np.allclose(friction,initial['geom_friction'],rtol=1e-7,atol=1e-9)
        manifest.update(initial_qpos_max_error_rad_or_m=qerror,initial_qvel_max_error=verror,
                        friction_verified=True,initial_state_key=f'{args.primitive}_trial{args.trial:02d}',
                        initial_qpos_sha256=__import__('hashlib').sha256(initial['qpos'].tobytes()).hexdigest(),
                        initial_friction_sha256=__import__('hashlib').sha256(initial['geom_friction'].tobytes()).hexdigest())
        robot=raw.scene[entity].data
        initial_com=robot.root_com_pos_w[0].detach().cpu().numpy().copy()
        manifest['initial_root_com']=initial_com.tolist()
        action_name='safe_joint2_residual' if args.primitive=='S01' else 'fourier_residual'
        term=raw.action_manager.get_term(action_name)
        obs=vec.get_observations()
        assert tuple(obs['actor'].shape)==(1,32) and raw.action_manager.total_action_dim==4
        manifest['status']='running'
        path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        for step in range(steps):
            if time.monotonic()-started>3600: raise TimeoutError('单条部署墙钟保护3600秒；不自动重试')
            if not all(bool(torch.isfinite(obs[g]).all()) for g in ('actor','critic')):
                raise FloatingPointError('策略观测非有限')
            with torch.inference_mode(): action=actor(obs['actor'])
            obs,_,dones,_=vec.step(action)
            state=capture(raw)
            if bool(dones[0]):
                manifest['reset_events'].append(step+1)
                np.savez_compressed(args.out/'post_reset_event_state.npz',**state)
                raise RuntimeError('部署发生自动reset；保留之前连续段，不拼接路线')
            contact_count=int(raw.sim.data.nacon[0].item())
            constraints=int(raw.sim.data.nefc.max().item())
            if contact_count>=raw.sim.wp_data.naconmax or constraints>=raw.sim.wp_data.njmax:
                raise RuntimeError('接触/约束达到工作区容量，禁止截断继续')
            manifest['max_contacts']=max(manifest.get('max_contacts',0),contact_count)
            manifest['max_constraints']=max(manifest.get('max_constraints',0),constraints)
            for name in FIELDS: trace[name].append(state[name])
            phase,reference=nominal_paper(args.primitive,term)
            qpaper=paper_angles(raw.sim.mj_model,state['qpos'])
            com=robot.root_com_pos_w[0].detach().cpu().numpy().copy()
            quat=state['qpos'][3:7]
            w,x,y,z=quat
            yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
            settled=True if command is None else bool(command.settled[0].item())
            a=action[0].detach().cpu().numpy().copy()
            actual_residual=term._bounded_residual[0].detach().cpu().numpy().copy()*term.cfg.residual_scale
            # 动作应用后的命令settled可能刚跨边界；空间实际目标差更准确。
            if args.primitive!='S01':
                actual_residual=(term._target[0]-term.nominal_target[0]).detach().cpu().numpy().copy()
            if args.controller=='nominal': assert np.all(actual_residual==0)
            rows.append([(step+1)*.008,*com,yaw,phase,int(settled),*qpaper,*reference])
            actions.append(a)
            residuals.append(actual_residual)
            manifest['completed_steps']=len(rows)
            if step%500==0 or step==steps-1:
                manifest['wall_seconds']=time.monotonic()-started
                path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
                print(json.dumps(dict(step=step+1,budget=steps,wall_seconds=manifest['wall_seconds'])),flush=True)
        check_sources()
        manifest['status']='precheck_complete' if args.precheck else 'complete'
    except Exception as caught:
        error=caught
        manifest.update(status='failed',error=repr(caught))
    finally:
        if raw is not None: raw.close()
        np.savez_compressed(args.out/'control_and_solver_trace.npz',**{k:np.asarray(v) for k,v in trace.items()},dt=.008)
        columns='time_s,com_x,com_y,com_z,yaw_rad,command_phase,settled,'
        columns+=','.join(f'q{i}_paper' for i in range(1,8))+','+','.join(f'ref{i}_paper' for i in range(1,8))
        np.savetxt(args.out/'trajectory.csv',rows,delimiter=',',header=columns,comments='')
        np.savetxt(args.out/'actions.csv',actions,delimiter=',',header='J2,J3,J7,J1',comments='')
        np.savetxt(args.out/'applied_residual_rad.csv',residuals,delimiter=',',header='J2,J3,J7,J1',comments='')
        manifest.update(completed_steps=len(rows),physical_duration_s=len(rows)*.008,
                        wall_seconds=time.monotonic()-started)
        path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if error is not None: raise error
    if not args.precheck:
        from replay_metrics import analyze_record
        analyze_record(args.out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--primitive',choices=('S01','S02','S03'),required=True)
    parser.add_argument('--controller',choices=('seed42','seed43','seed44','nominal'),required=True)
    parser.add_argument('--trial',type=int,choices=range(1,11),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--precheck',action='store_true')
    torch.set_num_threads(1)
    record(parser.parse_args())
