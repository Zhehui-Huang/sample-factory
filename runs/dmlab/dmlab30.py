from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000]),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' --with_wandb=True --wandb_project=stabilized-rl '
    '--wandb_group=pre_dmlab_sf_origin --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'pre_dmlab_sf_origin',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('pre_dmlab_sf_origin', experiments=[_experiment])
