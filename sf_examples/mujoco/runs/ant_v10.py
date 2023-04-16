from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from sf_examples.mujoco.runs.baseline import ANT_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111]),
    ('lr_schedule', ['kl_adaptive_minibatch']),
    ('start_target_kl', [0.15, 0.2]),
    ('start_kl_steps', [250000, 500000, 1000000]),
])

ANT_CLI = ANT_BASELINE_CLI + (
    ' --kl_loss_coeff_lr=0.1 --target_kl=0.1 --MIN_KL_LOSS_COEFF=0.001 --with_wandb=True --wandb_project=stabilized-rl '
    '--wandb_group=sf_ant_search_v10 --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'sf_ant_search_v10',
    ANT_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('sf_ant', experiments=[_experiment])