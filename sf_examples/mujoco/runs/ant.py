from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from sf_examples.mujoco.runs.baseline import ANT_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
    ('kl_loss_coeff_lr', [0.1, 0.2]),
    ('target_kl', [0.03, 0.05, 0.10]),
    ('MIN_KL_LOSS_COEFF', [0.001, 0.01]),
])

ANT_CLI = ANT_BASELINE_CLI + (
    ' --with_wandb=True --wandb_project=stabilized-rl '
    '--wandb_group=sf_ant_search --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'sf_ant_search',
    ANT_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('sf_ant', experiments=[_experiment])