from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from sf_examples.mujoco.runs.baseline import ANT_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
    ('lr_schedule', ['kl_adaptive_minibatch']),
])

ANT_CLI = ANT_BASELINE_CLI + (
    ' --kl_loss_coeff_lr=0.1 --target_kl=0.1 --MIN_KL_LOSS_COEFF=0.001 --with_wandb=True --wandb_project=stabilized-rl '
    '--wandb_group=sf_ant_search_v_11_adam --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'sf_ant_search_v11',
    ANT_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('sf_ant', experiments=[_experiment])
