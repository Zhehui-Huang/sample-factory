from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' --env=dmlab_collect_good_objects --with_wandb=True --wandb_project=stabilized-rl --train_for_env_steps=100000000 '
    '--wandb_group=dmlab_collect_good_objects_sf_origin_v3 --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'dmlab_collect_good_objects_sf_origin_v3',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('sf_ori_dmlab_collect_good_objects', experiments=[_experiment])
