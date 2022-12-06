from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
        (
            "env",
            [
                "mujoco_doublependulum",
                "mujoco_pendulum",
            ],
        ),
    ]
)

_experiments = [
    Experiment(
        "mujoco_all_envs",
        "python -m sample_factory_examples.mujoco_examples.train_mujoco --algo=APPO --with_wandb=True "
        "--wandb_user=multi-drones --wandb_project=zh-reward-decrease --wandb_tags mujoco pendulum 0b7c22ca4de0d5fc7f4f79e76afbfc7ef67ddd87 "
        "--train_for_env_steps=100000000",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("mujoco_all_envs", experiments=_experiments)
# python -m sample_factory.runner.run --run=sample_factory_examples.mujoco_examples.mujoco_pendulum --runner=processes --max_parallel=4  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1 --experiment_suffix=4
