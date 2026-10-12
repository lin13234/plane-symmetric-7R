"""在独立物理资产上从零训练 S03；不会修改旧任务、模型或奖励。

唯一任务名只改变记录名称。连续旋转、参考、奖励、命令采样和 PPO
预算保持当前配方；初始化证据在第一次优化前保存，任何加载均拒绝。
"""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json
import os
import torch
from mjlab.tasks.stage2_skill.config import SKILL_SPECS, make_stage2_env_cfg, make_stage2_ppo_cfg
from mjlab.rl import MjlabOnPolicyRunner

ROOT = Path(__file__).resolve().parent
TASK = 'stage2-s03-common-model-round2-20261012'
XML = ROOT / 'assets/zhengti_stage2_mesh_collision.xml'


def model_digest(state):
    """固定键序和原始数值摘要，用于区分随机 actor/critic。"""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FreshRunner(MjlabOnPolicyRunner):
    """完整随机初始化；拒绝 resume 和 actor-only 热启动。"""
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        if train_cfg.get('resume') or train_cfg.get('warm_start_actor'):
            raise ValueError('本次三个种子必须从零开始')
        super().__init__(env, train_cfg, log_dir, device)
        self.protocol = json.loads((ROOT/'protocol.json').read_text())
        # nacon 为所有环境的总尝试接触数；nefc 是每个环境的约束数。
        # 超容量时立即失败，不允许以截断约束继续训练。
        raw = env.unwrapped
        original_step = raw.step
        self.capacity_statistics = {'maximum_contacts_global':0,'maximum_constraints_per_environment':0,
                                    'nconmax_per_environment':256,'njmax_per_environment':1024,
                                    'overflow_events':0,'guarded_control_steps':0}
        def guarded_step(action):
            result = original_step(action)
            contacts = int(raw.sim.data.nacon[0].item())
            constraints = int(raw.sim.data.nefc.max().item())
            stats = self.capacity_statistics
            stats['maximum_contacts_global'] = max(stats['maximum_contacts_global'],contacts)
            stats['maximum_constraints_per_environment'] = max(stats['maximum_constraints_per_environment'],constraints)
            stats['guarded_control_steps'] += 1
            if contacts >= raw.sim.wp_data.naconmax or constraints >= raw.sim.wp_data.njmax:
                stats['overflow_events'] += 1
                (Path(log_dir)/'workspace_capacity_audit.json').write_text(json.dumps(stats,indent=2))
                raise RuntimeError('接触或约束达到分配容量，禁止截断后继续训练')
            return result
        raw.step = guarded_step
        initial = Path(log_dir)/'initial_model.pt'
        self.save(str(initial))
        cp = torch.load(initial, map_location='cpu', weights_only=False)
        if cp['iter'] != 0 or cp['optimizer_state_dict']['state']:
            raise ValueError('初始更新数或优化器状态不符合从零要求')
        props = torch.cuda.get_device_properties(torch.device(device))
        actual = str(props.uuid).lower().removeprefix('gpu-')
        expected = os.environ['CUDA_VISIBLE_DEVICES'].lower().removeprefix('gpu-')
        assert actual == expected, '设备 UUID 未正确隔离'
        evidence = dict(seed=train_cfg['seed'], updates_before_save=0,
                        optimizer_empty=True, resume=False, weights_loaded=False,
                        actor_sha256=model_digest(cp['actor_state_dict']),
                        critic_sha256=model_digest(cp['critic_state_dict']),
                        xml_sha256=hashlib.sha256(XML.read_bytes()).hexdigest(),
                        gpu_name=props.name, gpu_uuid=str(props.uuid))
        (Path(log_dir)/'initialization_audit.json').write_text(json.dumps(evidence,indent=2))
        print('[FRESH_INITIALIZATION_VERIFIED] '+json.dumps(evidence), flush=True)

    def save(self, path, infos=None):
        if hasattr(self,'capacity_statistics'):
            (Path(path).parent/'workspace_capacity_audit.json').write_text(json.dumps(self.capacity_statistics,indent=2))
        return super().save(path, infos={**(infos or {}), 'round2_s03':{
            'xml_sha256':hashlib.sha256(XML.read_bytes()).hexdigest(),
            'random_initialization':True, 'warm_start_actor':None,
            'training_and_deployment_xml':str(XML)}})

    def load(self, *args, **kwargs):
        raise RuntimeError('禁止加载旧 checkpoint 或重启同一训练')


def configuration(seed):
    """只将实体名和资产路径指向独立实验，科学参数保持不变。"""
    spec = replace(SKILL_SPECS['stage2-spatial-rotary'], task_id=TASK, xml_path=str(XML))
    cfg = make_stage2_env_cfg(spec)
    # 只调整存储工作区与线搜索执行方式；物理参数保持原样。
    cfg.sim.nconmax = 256
    cfg.sim.njmax = 1024
    cfg.sim.ls_parallel = False
    cfg.seed = seed
    cfg.scene.num_envs = 2048
    agent = make_stage2_ppo_cfg('Stage2_S03_CommonModel_Round2_20261012', seed)
    agent.resume = False
    agent.max_iterations = 800
    agent.logger = 'tensorboard'
    agent.upload_model = False
    return spec, cfg, agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(42,43,44), default=42)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--audit-config', action='store_true')
    args = parser.parse_args()
    spec, cfg, agent = configuration(args.seed)
    if args.audit_config:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(XML))
        result = dict(spec=asdict(spec), environments=cfg.scene.num_envs,
                      rollout_steps=agent.num_steps_per_env, updates=agent.max_iterations,
                      resume=agent.resume, residual_scale=cfg.actions['fourier_residual'].residual_scale,
                      command_directions=list(cfg.commands['skill'].direction_choices),
                      force_caps=model.actuator_forcerange.tolist(),
                      workspace=dict(nconmax=cfg.sim.nconmax,njmax=cfg.sim.njmax,
                                     ls_parallel=cfg.sim.ls_parallel),
                      total_mass=float(model.body_mass.sum()), xml_sha256=hashlib.sha256(XML.read_bytes()).hexdigest())
        assert result['spec']['forward_sign'] == 1.0
        assert result['spec']['phase']['speed'] == .8 and result['residual_scale'] == .05
        assert result['force_caps'] == [[-.68,.68],[-.68,.68],[-1.8,1.8],[-1.8,1.8]]
        (ROOT/'configuration_audit.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
        return
    if args.run_dir is None or args.run_dir.exists():
        raise ValueError('必须提供尚不存在的训练目录')
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.scripts.train import TrainConfig, run_train
    play = make_stage2_env_cfg(spec,play=True)
    play.sim.nconmax = 256
    play.sim.njmax = 1024
    play.sim.ls_parallel = False
    register_mjlab_task(TASK, cfg, play, agent, FreshRunner)
    run_train(TASK, TrainConfig(env=cfg,agent=agent,enable_nan_guard=True), args.run_dir)


if __name__ == '__main__':
    main()
