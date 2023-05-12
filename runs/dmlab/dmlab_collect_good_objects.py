from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
    ('target_kl', [0.02]),
    ('kl_loss_coeff_lr', [5.0]),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' --env=dmlab_collect_good_objects --MIN_KL_LOSS_COEFF=0.001 --start_kl_steps=0 --train_for_env_steps=100000000 '
    '--with_wandb=True --wandb_project=stabilized-rl --wandb_group=dmlab_collect_good_objects_sf_xppo_value_func --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'dmlab_collect_good_objects_sf_xppo_value_func',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('sf_xppo_dmlab_collect_good_objects', experiments=[_experiment])
