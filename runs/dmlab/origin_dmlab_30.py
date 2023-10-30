from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
    ('batch_size', [1024]),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' --env=dmlab_30 '
    '--train_for_env_steps=1000000000 --with_wandb=True --wandb_project=stabilized-rl '
    '--wandb_group=sf-origin_dmlab_30 --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'dmlab_30_v1',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('origin_sf', experiments=[_experiment])