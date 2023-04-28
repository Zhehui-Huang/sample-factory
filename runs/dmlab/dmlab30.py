from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111]),
    ('num_batches_per_epoch', [2, 4]),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' --target_kl=0.001 --kl_loss_coeff_lr=0.1 --MIN_KL_LOSS_COEFF=0.001 --start_kl_steps=0 '
    '--with_wandb=True --wandb_project=stabilized-rl --wandb_group=dmlab_sf_xppo_minibatch --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'dmlab_sf_xppo_minibatch',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('sf_xppo_dmlab', experiments=[_experiment])
