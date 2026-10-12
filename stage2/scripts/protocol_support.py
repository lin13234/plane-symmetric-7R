"""新增实验的公共边界：只读来源、独立配置、实际初态与设备核验。

此扩展不修改运行中的 S03 工厂。原有动作、观测、奖励函数直接复用；
明确修改项仅为已批准的部署时间/距离终止、配对初态、滑动摩擦与工作区。
"""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parent
TRAIN=ROOT.parent/'workspace_revision_v2'
HUB=ROOT.parent.parent
RUNTIME=Path(os.environ.get('MJLAB_RUNTIME', '.'))
sys.path[:0]=[str(HUB),str(RUNTIME/'src')]
GPUS=('GPU-69cd9cad-5fbe-e351-56d7-6d7d5f835ca5','GPU-c497c357-be94-39a8-3a6e-8d4e3bff76ce')
XMLS={'S01':HUB/'mjlab_stage2_joint2_rebuilt/src/mjlab/asset_zoo/stage2_joint2_rebuilt/zhengti_joint2_v2.xml',
      'S02':TRAIN/'assets/zhengti_stage2_mesh_collision.xml',
      'S03':TRAIN/'assets/zhengti_stage2_mesh_collision.xml',
      'StageI':RUNTIME/'src/mjlab/asset_zoo/classtrain1-2/zhengti.xml'}


def sha(path):
    """摘要计算只读取文件，不覆盖来源清单。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources():
    """训练原清单与新增扩展清单均须一致，禁止重新计算以隐藏变化。"""
    for manifest in (TRAIN/'source_before.json',ROOT/'source_before.json'):
        for name,value in json.loads(manifest.read_text()).items():
            if sha(name)!=value: raise RuntimeError('冻结来源变化：'+name)


def bootstrap_s01():
    """把既有独立 J2 配置加到当前进程的包搜索目录，不改共享文件。"""
    import mjlab.tasks
    overlay=str(HUB/'mjlab_stage2_joint2_rebuilt/src/mjlab/tasks')
    if overlay not in mjlab.tasks.__path__: mjlab.tasks.__path__.append(overlay)


def device_identity():
    """CUDA 0 只表示按完整 UUID 隔离后的唯一设备。"""
    import torch
    expected=os.environ.get('CUDA_VISIBLE_DEVICES')
    if expected not in GPUS: raise RuntimeError('必须按已批准的完整 GPU UUID 隔离')
    props=torch.cuda.get_device_properties(0)
    assert str(props.uuid).lower().removeprefix('gpu-')==expected.lower().removeprefix('gpu-')
    return dict(uuid=expected,name=props.name)


def reset_to_full_state(env,env_ids,entity,qpos,qvel):
    """默认 reset 后仅写实际姿态/速度，默认动作/观测偏置保持不变。"""
    import torch
    from mjlab.envs.mdp.events import reset_scene_to_default
    reset_scene_to_default(env,env_ids)
    if env_ids is None: env_ids=torch.arange(env.num_envs,device=env.device)
    count=len(env_ids)
    asset=env.scene[entity]
    fullq=torch.as_tensor(qpos,dtype=torch.float32,device=env.device)
    fullv=torch.as_tensor(qvel,dtype=torch.float32,device=env.device)
    if len(qpos)==14 and len(qvel)==13:
        root=torch.cat((fullq[:7],fullv[:6])).repeat(count,1)
        asset.write_root_state_to_sim(root,env_ids=env_ids)
        asset.write_joint_state_to_sim(fullq[7:].repeat(count,1),fullv[6:].repeat(count,1),env_ids=env_ids)
    elif len(qpos)==len(qvel)==7:
        asset.write_joint_state_to_sim(fullq.repeat(count,1),fullv.repeat(count,1),env_ids=env_ids)
    else: raise ValueError('实际初态维度不符合预审模型')


def paired_config(primitive,trial):
    """复用三种原工厂；只有配对列表中的初态和滑动摩擦不同。"""
    import csv
    import numpy as np
    import mujoco
    from mjlab.managers import EventTermCfg
    if primitive=='S01':
        bootstrap_s01()
        from stage2_s01_j2_obs32.task import make_env_cfg
        cfg=make_env_cfg(play=True)
        entity=next(iter(cfg.scene.entities))
    elif primitive=='S02':
        from stage2_inchworm_single_side.task import make_env_cfg
        cfg=make_env_cfg(play=True)
        entity=next(iter(cfg.scene.entities))
    elif primitive=='S03':
        from mjlab.tasks.stage2_skill.config import SKILL_SPECS,make_stage2_env_cfg
        spec=replace(SKILL_SPECS['stage2-spatial-rotary'],xml_path=str(XMLS['S03']))
        cfg=make_stage2_env_cfg(spec,play=True)
        entity=spec.task_id
    else: raise ValueError(primitive)
    rows=list(csv.DictReader((TRAIN/'paired_perturbations.csv').open()))
    row=next(r for r in rows if r['primitive']==primitive and int(r['trial'])==trial)
    factor=float(row['sliding_friction_multiplier'])
    def spec_with_friction():
        # 只缩放实际参与接触的几何体滑动摩擦，其他参数从原 XML 读取。
        spec=mujoco.MjSpec.from_file(str(XMLS[primitive]))
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity: geom.friction[0]*=factor
        return spec
    cfg.scene.entities[entity].spec_fn=spec_with_friction
    data=np.load(TRAIN/'prepared_initial_states.npz')
    key=f'{primitive}_trial{trial:02d}'
    qpos,qvel=data[key+'_qpos'],data[key+'_qvel']
    cfg.events={'paired_reset':EventTermCfg(func=reset_to_full_state,mode='reset',
                params=dict(entity=entity,qpos=qpos.tolist(),qvel=qvel.tolist()))}
    cfg.seed=42
    cfg.scene.num_envs=1
    seconds=42 if primitive=='S02' else 64
    cfg.episode_length_s=2*seconds
    # 与既有64s协议一致，取消5m保护；保留其他终止且记录全部reset。
    cfg.terminations.pop('distance_safety',None)
    cfg.sim.nconmax=256
    cfg.sim.njmax=1024
    cfg.sim.ls_parallel=False
    return cfg,entity,qpos,qvel,row


def checkpoint_path(primitive,controller):
    """正式 S03 必须是本次三份最终模型，不能以中期模型或旧组替代。"""
    if controller=='nominal': return None
    seed=int(controller.removeprefix('seed'))
    if seed not in (42,43,44): raise ValueError(controller)
    if primitive=='S01':
        return HUB/f'stage2_current_archive_20261010/S01/seed{seed}/training/model_799.pt'
    if primitive=='S02': return HUB/f'stage2_s02_fresh_20261009/seed{seed}/run/model_799.pt'
    audit=TRAIN/f'seed{seed}/final_checkpoint_audit.json'
    evidence=json.loads(audit.read_text())
    assert evidence['completed_updates']==800 and evidence['finite_weights']
    return TRAIN/f'seed{seed}/run/model_799.pt'


def actor_mean(checkpoint,device):
    """32→256→128→4 ELU均值网络；不加载优化器或采样方差。"""
    import torch
    if checkpoint is None: return lambda obs: torch.zeros((obs.shape[0],4),device=obs.device)
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert cp['iter']==799
    state=cp['actor_state_dict']
    expected={'mlp.0.weight':(256,32),'mlp.2.weight':(128,256),'mlp.4.weight':(4,128)}
    assert all(tuple(state[k].shape)==shape for k,shape in expected.items())
    # 原配置关闭观测归一化；拒绝出现需要另行重建的归一化/RNN状态。
    extras=[k for k in state if not k.startswith('mlp.') and not k.startswith('distribution.')]
    if extras: raise ValueError('存在未知非均值网络状态：'+str(extras))
    assert all(bool(torch.isfinite(v).all()) for v in state.values())
    model=torch.nn.Sequential(torch.nn.Linear(32,256),torch.nn.ELU(),
                             torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,4))
    model.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')},strict=True)
    return model.to(device).eval()


def paper_angles(model,qpos):
    """按模型的实际关节标签查地址，J4/J7注册符号翻转。"""
    import numpy as np
    import mujoco
    labels={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i).rsplit('/',1)[-1]:i for i in range(model.njnt)}
    addresses=[int(model.jnt_qposadr[labels[f'joint{i}']]) for i in range(1,8)]
    angles=np.asarray(qpos)[...,addresses].copy()
    angles[..., [3,6]]*=-1
    return angles
