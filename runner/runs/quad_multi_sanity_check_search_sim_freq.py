from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [0000, 1111, 2222]),
    ('quads_num_agents', [4]),
    ('quads_episode_duration', [7.0]),
    ('quads_mode', ['sanity_check']),
    ('quads_sim_freq', [100, 200]),
    ('quads_sim_steps', [2, 4]),
])

_experiment = Experiment(
    'search_sim_and_control_freq',
    'python -m run_algorithm --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False --num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --adam_eps=1e-8 --ppo_clip_value=5.0 --recurrence=1 --nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False --hidden_size=64 --with_vtrace=False --max_policy_lag=100000000 --gae_lambda=1.00 --max_grad_norm=0.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=512 --extend_obs=True --quads_use_numba=True --quads_dist_between_goals=0.0 --quads_collision_reward=0.0',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_sanity_check_v112', experiments=[_experiment])

# Note: Before using this script and running the script, I would recommend you
# reading the spreadsheet, page: sanity_check, first.

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_sanity_check_search_sim_freq --runner=processes --max_parallel=12 --pause_between=1 --experiments_per_gpu=3 --num_gpus=4
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_sanity_check_search_sim_freq --runner=processes --max_parallel=12 --pause_between=1 --experiments_per_gpu=3 --num_gpus=4
