"""只注册新的S01-32D任务，复用正确J2物理/奖励/动作，不覆盖14维版。"""
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.stage2_joint2_rebuilt.config import TASK_ID as ENTITY_NAME
from mjlab.tasks.stage2_joint2_rebuilt.config import make_joint2_env_cfg, make_joint2_ppo_cfg
from .observations import root_velocity, tracking_error, phase_command

TASK_ID = 'stage2-joint2-planar-residual-obs32-v3'
EXPERIMENT_NAME = 'Stage2_Joint2_Planar_Residual_Obs32_V3'
OBSERVATION_LAYOUT = {'joint_pos':7, 'joint_vel':7, 'root_velocity':4, 'tracking_error':7, 'phase_command':7}


def make_env_cfg(play=False):
    """仅增加观测，并显式冻结已成功14维训练的动作及运行参数。"""
    cfg = make_joint2_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 512
    action = cfg.actions['safe_joint2_residual']
    action.residual_scale = .02
    action.period_s = 4.
    action.safe_lower, action.safe_upper = -1.45, 2.25
    robot_cfg = SceneEntityCfg(ENTITY_NAME)
    # 原工厂actor/critic共享terms对象，因此分别复制字典，避免重复或相互污染。
    for group in ('actor', 'critic'):
        terms = dict(cfg.observations[group].terms)
        for name, fn in (('root_velocity', root_velocity), ('tracking_error', tracking_error), ('phase_command', phase_command)):
            terms[name] = ObservationTermCfg(func=fn, params={'robot_cfg':robot_cfg})
        cfg.observations[group].terms = terms
    return cfg


def make_rl_cfg(seed=42):
    """每个seed独立随机初始化；保留成功J2基线的PPO配方而非旧错误S01。"""
    if seed not in (42, 43, 44):
        raise ValueError('本次授权seed为42、43、44')
    cfg = make_joint2_ppo_cfg(seed)
    cfg.experiment_name = EXPERIMENT_NAME
    cfg.max_iterations = 800
    cfg.save_interval = 25
    cfg.logger = 'tensorboard'
    cfg.upload_model = False
    cfg.resume = False
    cfg.run_name = f's01_j2_obs32_seed{seed}_800_20261002'
    return cfg


def register_task(runner_cls=None):
    """幂等新增任务；同名异构注册直接拒绝，不去修改共享注册表。"""
    from mjlab.tasks.registry import list_tasks, load_env_cfg, register_mjlab_task
    if TASK_ID in list_tasks():
        cfg = load_env_cfg(TASK_ID)
        if tuple(cfg.observations['actor'].terms) != tuple(OBSERVATION_LAYOUT):
            raise ValueError('同名任务存在不同观测定义')
        return
    register_mjlab_task(TASK_ID, make_env_cfg(), make_env_cfg(play=True), make_rl_cfg(), runner_cls=runner_cls)
