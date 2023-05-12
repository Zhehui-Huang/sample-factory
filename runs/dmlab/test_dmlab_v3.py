from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111]),
    ('env', ['dmlab_skymaze_irreversible_path_hard', 'dmlab_skymaze_irreversible_path_varied',
             'dmlab_explore_object_locations_small', 'dmlab_explore_object_locations_large']),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' --num_workers=16 --num_envs_per_worker=12 --dmlab_one_task_per_worker=False --with_wandb=True '
    '--wandb_project=stabilized-rl --train_for_env_steps=100000000 '
    '--wandb_group=test_dmlab --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'test_dmlab',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('test_dmlab', experiments=[_experiment])
