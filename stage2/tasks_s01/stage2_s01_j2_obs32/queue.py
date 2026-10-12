"""新S01隔离串行队列：42→43→44，后台脱离终端，失败不连带其它训练。"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HUB = Path(os.environ.get('DATA_ROOT', '.'))
ROOT = HUB / 'stage2_s01_j2_obs32_training_20261002'
REBUILT = HUB / 'mjlab_stage2_joint2_rebuilt/src'
PYTHON = os.environ.get('PYTHON_BIN', 'python')
EXPERIMENT = 'Stage2_Joint2_Planar_Residual_Obs32_V3'
GPU_UUID = 'GPU-69cd9cad-5fbe-e351-56d7-6d7d5f835ca5'


def training_command(seed):
    """每个进程只指定自己的seed，不加载14维、旧S01或前一个seed权重。"""
    if seed not in (42,43,44):
        raise ValueError('非授权种子')
    return [PYTHON, '-m', 'stage2_s01_j2_obs32.train', '--seed', str(seed)]


def child_environment():
    """设备映射只作用于本队列子进程，绝不改变已有S02环境。"""
    env = dict(os.environ)
    for key in ('CUDA_VISIBLE_DEVICES', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE'):
        env.pop(key, None)
    env.update(PYTHONPATH=f'{HUB}:{REBUILT}', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
    return env


def gpu_is_available(snapshot):
    """只在5060Ti持续低负载且有至少8GiB余量时启动；不与未知占用抢卡。"""
    return snapshot['utilization_percent'] <= 20 and snapshot['free_mib'] >= 8192


def parse_gpu_snapshot(raw):
    """按UUID识别指定物理卡，避免把WSL CUDA顺序误认为nvidia-smi编号。"""
    values = [v.strip() for v in raw.strip().split(',')]
    if len(values) != 5 or values[0] != GPU_UUID or '5060 Ti' not in values[1]:
        raise ValueError(f'目标显卡身份不符：{raw!r}')
    # 不同驱动的CSV默认带MiB/%；既支持带单位样本，也支持nounits输出。
    used, total, utilization = (int(v.split()[0]) for v in values[2:])
    if not (0 <= used <= total and 0 <= utilization <= 100):
        raise ValueError('显卡状态数值非法')
    return {'uuid':values[0], 'name':values[1], 'used_mib':used, 'total_mib':total,
            'free_mib':total-used, 'utilization_percent':utilization}


def wait_for_gpu(state, stage):
    """后台每30秒探测，连续3次空闲才放行；等待不创建CUDA上下文。"""
    consecutive = 0
    state.update(status='waiting_for_gpu', waiting_stage=stage)
    while consecutive < 3:
        raw = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', f'--id={GPU_UUID}',
            '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=15)
        state['gpu_snapshot'] = parse_gpu_snapshot(raw)
        consecutive = consecutive+1 if gpu_is_available(state['gpu_snapshot']) else 0
        state.update(idle_samples=consecutive, updated_at=datetime.now().astimezone().isoformat())
        write_json(ROOT/'status.json', state)
        if consecutive < 3:
            time.sleep(30)


def run_smoke(state, baseline):
    """空卡后先完成同物理/同接口16环境两次PPO，未通过绝不启动正式800轮。"""
    wait_for_gpu(state, 'smoke_16env_2updates')
    if source_hashes() != baseline:
        raise ValueError('等待期间源文件发生变化，不能自动训练')
    folder = ROOT/'smoke'
    folder.mkdir()
    command = training_command(42)+['--smoke']
    with (folder/'train.log').open('w') as output:
        process = subprocess.Popen(command, cwd=HUB, env=child_environment(), stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT)
        state.update(status='running_smoke', smoke_pid=process.pid, active_seed=None)
        write_json(ROOT/'status.json', state)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f'冒烟退出码{code}；正式seed均不启动')
    runs = list((HUB/'logs/rsl_rl'/(EXPERIMENT+'_Smoke')).glob('*_smoke_seed42_20261002'))
    if len(runs) != 1:
        raise ValueError('冒烟目录不唯一')
    audit = audit_run(runs[0], expected_iteration=1)
    write_json(folder/'completion_audit.json', audit)
    state.update(smoke_status='passed', smoke_run_dir=str(runs[0]))
    write_json(ROOT/'status.json', state)


def source_hashes():
    """冻结新接口和所复用物理/参考/训练库；JSON日志/pycache不进入源码清单。"""
    files = set((HUB/'stage2_s01_j2_obs32').glob('*.py'))
    files.update((REBUILT/'mjlab/tasks/stage2_joint2_rebuilt').glob('*.py'))
    files.update(p for p in (REBUILT/'mjlab/asset_zoo/stage2_joint2_rebuilt').rglob('*') if p.is_file())
    for rel in ('mjlab/scripts/train.py','mjlab/rl/runner.py','mjlab/rl/vecenv_wrapper.py',
                'mjlab/envs/manager_based_rl_env.py','mjlab/envs/mdp/observations.py',
                'mjlab/managers/observation_manager.py'):
        files.add(REBUILT/rel)
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def write_json(path, data):
    """原子更新本实验状态，避免审核时读到半写入JSON。"""
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def validate_preflight_retry(state, root):
    """只允许修复启动探测失败后重启；一旦冒烟/正式任务启动，不可借此重复训练。"""
    if state.get('status') != 'failed_pending_review' or (root/'smoke').exists():
        raise ValueError('只能重试从未启动冒烟的失败队列')
    seeds = state.get('seeds', {})
    if set(seeds) != {'42','43','44'} or any(job.get('status') != 'queued' or 'pid' in job for job in seeds.values()):
        raise ValueError('seed已经启动或记录不完整，禁止重试覆盖')


def validate_checkpoint(cp, expected_iteration):
    """完成必须同时有正确更新编号、actor/critic形状和有限浮点权重。"""
    import torch
    if cp['iter'] != expected_iteration:
        raise ValueError('checkpoint更新编号不符')
    for key, output in (('actor_state_dict',4), ('critic_state_dict',1)):
        state = cp[key]
        if tuple(state['mlp.0.weight'].shape) != (256,32) or tuple(state['mlp.4.weight'].shape) != (output,128):
            raise ValueError(f'{key}不符合32-256-128-{output}')
        if not all(torch.isfinite(v).all().item() for v in state.values() if torch.is_tensor(v)):
            raise ValueError(f'{key}含非有限权重')


def audit_run(run_dir, expected_iteration=799):
    """完成审计既检查checkpoint，也检查真实TensorBoard更新记录及冻结配方。"""
    import torch
    import yaml
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    cp_path = run_dir / f'model_{expected_iteration}.pt'
    cp = torch.load(cp_path, map_location='cpu', weights_only=False)
    validate_checkpoint(cp, expected_iteration)
    events = EventAccumulator(str(run_dir), size_guidance={'scalars':0}).Reload()
    rewards = events.Scalars('Train/mean_reward')
    expected = set(range(expected_iteration+1))
    if len(rewards) != len(expected) or {e.step for e in rewards} != expected:
        raise ValueError('奖励更新记录不完整或重复，不能宣称800轮完成')
    for tag in events.Tags()['scalars']:
        if not all(math.isfinite(e.value) for e in events.Scalars(tag)):
            raise ValueError(f'TensorBoard {tag}存在非有限值')
    agent = yaml.load((run_dir/'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    if agent['resume'] != 'false' or int(agent['max_iterations']) != expected_iteration+1:
        raise ValueError('训练不是独立随机初始化或迭代预算不符')
    protocol = json.loads((run_dir/'runtime_protocol.json').read_text())
    if '5060 Ti' not in protocol['gpu_name']:
        raise ValueError('实际GPU与指定设备不符')
    return {'checkpoint':str(cp_path), 'sha256':hashlib.sha256(cp_path.read_bytes()).hexdigest(),
            'completed_updates':expected_iteration+1, 'reward_entries':len(rewards),
            'final_reward':rewards[-1].value, 'protocol':protocol,
            'claim':'training completed; gait quality requires deterministic replay'}


def main():
    """单实例锁、独立日志及逐seed审计；异常只使本队列失败，不自动重启。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--retry-preflight', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if args.detach:
        if (ROOT/'status.json').exists():
            if not args.retry_preflight:
                raise RuntimeError('已有队列状态，禁止重复启动')
            # 对自己的失败队列取排他锁；归档而非删除原失败证据。
            import fcntl
            with (ROOT/'queue.lock').open('a') as old_lock:
                fcntl.flock(old_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                validate_preflight_retry(json.loads((ROOT/'status.json').read_text()), ROOT)
                suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
                (ROOT/'status.json').rename(ROOT/f'failed_preflight_{suffix}.json')
                (ROOT/'source_before.json').rename(ROOT/f'source_failed_preflight_{suffix}.json')
        with (ROOT/'queue.log').open('a') as out:
            process = subprocess.Popen([PYTHON,'-m','stage2_s01_j2_obs32.queue'], cwd=HUB,
                env=child_environment(), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT/'queue_pid.txt').write_text(str(process.pid))
        print(json.dumps({'queue_pid':process.pid,'root':str(ROOT)}), flush=True)
        return
    import fcntl
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'status.json').exists():
            raise RuntimeError('队列已存在，不自动重跑/覆盖')
        baseline = source_hashes()
        print(f'新队列启动 pid={os.getpid()} time={datetime.now().astimezone().isoformat()}', flush=True)
        write_json(ROOT/'source_before.json', baseline)
        state = {'queue_pid':os.getpid(), 'status':'starting', 'seeds':{str(s):{'status':'queued'} for s in (42,43,44)},
                 'target_updates_per_seed':800, 'active_seed':None,
                 'gpu':'physical0 RTX5060Ti / unmasked WSL CUDA1'}
        write_json(ROOT/'status.json',state)
        try:
            run_smoke(state, baseline)
            for seed in (42,43,44):
                wait_for_gpu(state, f'formal_seed{seed}')
                if source_hashes() != baseline:
                    raise ValueError('冻结源码/物理发生变化；停止本实验队列')
                job = state['seeds'][str(seed)]
                job.update(status='starting', started_at=datetime.now().astimezone().isoformat())
                log_dir = ROOT / f'seed{seed}'
                log_dir.mkdir()
                with (log_dir/'train.log').open('w') as output:
                    process = subprocess.Popen(training_command(seed), cwd=HUB, env=child_environment(),
                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
                    job.update(pid=process.pid, command=training_command(seed), status='running')
                    state.update(status='running', active_seed=seed)
                    write_json(ROOT/'status.json',state)
                    while process.poll() is None:
                        runs = list((HUB/'logs/rsl_rl'/EXPERIMENT).glob(f'*_s01_j2_obs32_seed{seed}_800_20261002'))
                        if len(runs) == 1:
                            job['run_dir'] = str(runs[0])
                            (log_dir/'run_path.txt').write_text(str(runs[0]))
                            checkpoints = list(runs[0].glob('model_*.pt'))
                            if checkpoints:
                                job['latest_saved_iteration'] = max(int(p.stem.split('_')[1]) for p in checkpoints)
                        state['updated_at'] = datetime.now().astimezone().isoformat()
                        write_json(ROOT/'status.json',state)
                        time.sleep(20)
                job['exit_code'] = process.returncode
                if process.returncode != 0:
                    raise RuntimeError(f'seed{seed}训练失败，退出码{process.returncode}；后续seed不启动')
                if source_hashes() != baseline:
                    raise ValueError('训练过程中源文件改变')
                runs = list((HUB/'logs/rsl_rl'/EXPERIMENT).glob(f'*_s01_j2_obs32_seed{seed}_800_20261002'))
                if len(runs) != 1:
                    raise ValueError('不能唯一识别本seed训练目录')
                audit = audit_run(runs[0])
                write_json(log_dir/'completion_audit.json', audit)
                job.update(status='completed', completed_updates=800, final_checkpoint=audit['checkpoint'])
                write_json(ROOT/'status.json',state)
            hashes = [json.loads((ROOT/f'seed{s}/completion_audit.json').read_text())['protocol']['initial_actor_sha256'] for s in (42,43,44)]
            if len(set(hashes)) != 3:
                raise ValueError('三种子初始网络相同，不满足独立初始化')
            write_json(ROOT/'source_after.json',source_hashes())
            state.update(status='all_training_completed_pending_replay', active_seed=None)
            write_json(ROOT/'status.json',state)
        except Exception as error:
            state.update(status='failed_pending_review', error=repr(error))
            write_json(ROOT/'status.json',state)
            raise


if __name__ == '__main__':
    main()
