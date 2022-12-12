from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
        (
            "env",
            [
                "mujoco_doublependulum",
            ],
        ),
    ]
)

_experiments = [
    Experiment(
        "mujoco_all_envs",
        "python -m sf_examples.mujoco.train_mujoco --algo=APPO --with_wandb=True "
        "--wandb_user=multi-drones --wandb_project=zh-reward-decrease --wandb_tags mujoco pendulum "
        "--train_for_env_steps=100000000 --serial_mode=True --wandb_job_type=SF_serial",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("mujoco_all_envs", experiments=_experiments)
# python -m sample_factory.launcher.run --run=sf_examples.mujoco.experiments.mujoco_all_envs --backend=processes --max_parallel=4  --pause_between=1 --experiments_per_gpu=4 --num_gpus=1
