from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111]),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' ----num_workers=30 --dmlab_one_task_per_worker=False --env=dmlab_sparse_doors --with_wandb=True --wandb_project=stabilized-rl '
    '--wandb_group=test_dmlab_door_sparse --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'test_dmlab_door_sparse',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('test_dmlab_door_sparse', experiments=[_experiment])
