from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('with_wandb', [True]),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' --wandb_project=stabilized-rl '
    '--wandb_group=dmlab_sf_origin_pbt --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'dmlab_sf_origin_pbt',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('sf_ori_dmlab_pbt', experiments=[_experiment])