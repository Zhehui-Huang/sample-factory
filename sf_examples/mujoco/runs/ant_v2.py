from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from sf_examples.mujoco.runs.baseline import ANT_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
    ('lr_schedule', ['kl_adaptive_minibatch', 'kl_adaptive_epoch']),
])

ANT_CLI = ANT_BASELINE_CLI + (
    ' --with_wandb=True --wandb_project=stabilized-rl '
    '--wandb_group=sf_ori_ant_v2 --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'sf_ori_ant_v2',
    ANT_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('original_ant', experiments=[_experiment])