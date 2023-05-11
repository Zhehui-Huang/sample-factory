from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111]),
    ('target_kl', [0.02]),
    ('kl_loss_coeff_lr', [5.0]),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' --num_workers=30 --dmlab_one_task_per_worker=False --env=dmlab_sparse_doors --MIN_KL_LOSS_COEFF=0.001 --start_kl_steps=0 '
    '--with_wandb=True --wandb_project=stabilized-rl --wandb_group=test_dmlab_door_sparse_sf_xppo_value_func --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'test_dmlab_door_sparse_sf_xppo_value_func',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('test_sf_xppo_dmlab_door_sparse', experiments=[_experiment])
