DMLAB30_BASELINE_CLI = (
    'python -m sf_examples.dmlab_examples.train_dmlab --env=dmlab_30 --train_for_env_steps=1000000000 --gamma=0.99 '
    '--use_rnn=True --num_workers=72 --num_envs_per_worker=12 --num_epochs=1 --rollout=32 --recurrence=32 '
    '--batch_size=2048 --benchmark=False --max_grad_norm=0.0 --dmlab_renderer=software '
    '--decorrelate_experience_max_seconds=120 --reset_timeout_seconds=300 --encoder_conv_architecture=resnet_impala '
    '--encoder_conv_mlp_layers=512 --nonlinearity=relu --rnn_type=lstm --dmlab_extended_action_set=True '
    '--num_policies=1 --experiment=dmlab_watermaze_resnet_w90_v12 --set_workers_cpu_affinity=True --max_policy_lag=35 '
    '--dmlab30_dataset=/home/zhehui/mixppo/dmlab/datasets/brady_konkle_oliva2008 --dmlab_use_level_cache=True '
    '--dmlab_level_cache_path=/home/zhehui/mixppo/dmlab/dmlab_cache'
)
