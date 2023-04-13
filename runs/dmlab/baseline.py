DMLAB30_BASELINE_CLI = (
    'python -m sf_examples.dmlab.train_dmlab --env=dmlab_30 --train_for_env_steps=1000000000 --gamma=0.99 '
    '--use_rnn=True --num_workers=90 --num_envs_per_worker=12 --num_epochs=1 --rollout=32 --recurrence=32 '
    '--batch_size=2048 --benchmark=False --max_grad_norm=0.0 --dmlab_renderer=software '
    '--decorrelate_experience_max_seconds=120 --encoder_conv_architecture=resnet_impala '
    '--encoder_conv_mlp_layers=512 --nonlinearity=relu --rnn_type=lstm --dmlab_extended_action_set=True '
    '--num_policies=1 --experiment=sf_ori_dmlab_30 --set_workers_cpu_affinity=True --max_policy_lag=35 '
    ' --dmlab30_dataset=/home/zhehui/mixppo/dmlab/datasets/brady_konkle_oliva2008 --dmlab_use_level_cache=True '
    '--dmlab_level_cache_path=/home/zhehui/mixppo/dmlab/.dmlab_cache'
)

DMLAB30_PBT_BASELINE_CLI = (
    'python -m sf_examples.dmlab.train_dmlab --env=dmlab_30 --train_for_env_steps=10000000000 --gamma=0.99 '
    '--use_rnn=True --num_workers=72 --num_envs_per_worker=12 --num_epochs=1 --rollout=32 --recurrence=32 '
    '--batch_size=2048 --benchmark=False --max_grad_norm=0.0 --dmlab_renderer=software '
    '--decorrelate_experience_max_seconds=120 --encoder_conv_architecture=resnet_impala '
    '--encoder_conv_mlp_layers=512 --nonlinearity=relu --rnn_type=lstm --dmlab_extended_action_set=True '
    '--num_policies=4 --pbt_replace_reward_gap=0.05 --pbt_replace_reward_gap_absolute=5.0 '
    '--pbt_period_env_steps=10000000 --pbt_start_mutation=100000000 --with_pbt=True '
    '--experiment=dmlab_30_resnet_4pbt --dmlab_one_task_per_worker=True --set_workers_cpu_affinity=True '
    '--max_policy_lag=35 --pbt_target_objective=dmlab_target_objective '
    '--dmlab30_dataset=/home/zhehui/mixppo/dmlab/datasets/brady_konkle_oliva2008 --dmlab_use_level_cache=True '
    '--dmlab_level_cache_path=/home/zhehui/mixppo/dmlab/dmlab_cache'
)
