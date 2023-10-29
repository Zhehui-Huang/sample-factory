from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from runs.dmlab.baseline import DMLAB30_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
    ('batch_size', [1024]),
    ('eps_kl', [0.1, 0.2, 0.5]),
])

DMLAB30_CLI = DMLAB30_BASELINE_CLI + (
    ' --env=dmlab_collect_good_objects --lock_beta_optim=True --beta_lr=0.1 --target_coeff=10.0 '
    '--train_for_env_steps=100000000 --with_wandb=True --wandb_project=stabilized-rl '
    '--wandb_group=sf-fixpo_dmlab_collect_good_objects --wandb_user=resl-mixppo'
)

_experiment = Experiment(
    'fixpo_dmlab_collect_good_objects',
    DMLAB30_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('eps_kl_v3', experiments=[_experiment])