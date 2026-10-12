"""新S01三种子的冻结训练入口；只允许指定5060Ti和已确认控制配方。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner
from .task import TASK_ID, OBSERVATION_LAYOUT, make_env_cfg, make_rl_cfg, register_task


class CheckedRunner(MjlabOnPolicyRunner):
    """在首轮PPO前核实真实GPU/观测，保存随机初始化证据；不加载任何旧模型。"""
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        gpu = torch.cuda.get_device_name(torch.device(device))
        if '5060 Ti' not in gpu:
            raise ValueError(f'本实验必须使用空闲5060Ti，不得抢占S02/Ada；实际{gpu}')
        obs = env.get_observations()
        for group in ('actor', 'critic'):
            if obs[group].shape != (env.num_envs, 32) or not torch.isfinite(obs[group]).all():
                raise ValueError(f'{group}观测不是有限32维')
        if env.num_actions != 4:
            raise ValueError('动作必须为四电机残差')
        super().__init__(env, train_cfg, log_dir, device)
        # 用当前运行库真正保存一份初始化checkpoint；正式训练的model0已是首轮更新后。
        initial_path = Path(log_dir) / 'initial_model.pt'
        self.save(str(initial_path))
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)
        from .queue import validate_checkpoint
        validate_checkpoint(initial, 0)
        actor_sha = hashlib.sha256()
        for name, value in sorted(initial['actor_state_dict'].items()):
            actor_sha.update(name.encode())
            actor_sha.update(value.contiguous().numpy().tobytes())
        evidence = {'seed':train_cfg['seed'], 'gpu_name':gpu, 'device':str(device),
                    'actor_shape':[32, 256, 128, 4], 'critic_shape':[32, 256, 128, 1],
                    'observation_layout':OBSERVATION_LAYOUT, 'initial_actor_sha256':actor_sha.hexdigest(),
                    'initialization':'fresh random actor/critic/optimizer; no checkpoint loaded',
                    'fixed_command':{'direction':1, 'speed_scale':1., 'stop':False, 'desired_speed_m_s':.05},
                    'period_s':4., 'residual_bound_rad':.02, 'j2_reference_window_rad':[-1.45,2.25],
                    'episode_length_s':26., 'no_new_settling_gate':True}
        (Path(log_dir) / 'runtime_protocol.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        print('[S01-32D verified] '+json.dumps(evidence), flush=True)


def main():
    """外部不提供可误覆写的控制参数；冒烟与正式训练使用同一物理及32维接口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(42,43,44))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    from mjlab.scripts.train import TrainConfig, launch_training
    register_task(CheckedRunner)
    env, agent = make_env_cfg(), make_rl_cfg(args.seed)
    if args.smoke:
        env.scene.num_envs = 16
        agent.max_iterations = 2
        agent.save_interval = 1
        agent.experiment_name += '_Smoke'
        agent.run_name = f'smoke_seed{args.seed}_20261002'
    # select_gpus在入口把未屏蔽CUDA1映射为进程内CUDA0；不要预设CUDA_VISIBLE_DEVICES。
    cfg = TrainConfig(env=env, agent=agent, gpu_ids=[1], enable_nan_guard=True)
    launch_training(TASK_ID, cfg)


if __name__ == '__main__':
    main()
