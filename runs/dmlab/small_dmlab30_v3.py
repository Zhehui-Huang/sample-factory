from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import SMALL_NUM_ENV_DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000]),
    ('target_kl', [0.001, 0.01, 0.05, 0.1]),
])

DMLAB30_CLI = SMALL_NUM_ENV_DMLAB30_BASELINE_CLI + (
    ' --kl_loss_coeff_lr=3.0 --MIN_KL_LOSS_COEFF=0.001 --start_kl_steps=0 '
    '--with_wandb=True --wandb_project=stabilized-rl --wandb_group=small_env_dmlab_sf_xppo --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'small_dmlab_sf_xppo_target_kl_search',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('sf_xppo_dmlab', experiments=[_experiment])
