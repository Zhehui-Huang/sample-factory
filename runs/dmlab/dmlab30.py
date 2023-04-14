from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' --with_wandb=True --wandb_project=stabilized-rl '
    '--wandb_group=dmlab_sf_origin_v2 --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'dmlab_sf_origin_v2',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('sf_ori_dmlab', experiments=[_experiment])
