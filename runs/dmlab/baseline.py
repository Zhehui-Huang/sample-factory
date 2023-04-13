DMLAB30_BASELINE_CLI = (
    'python -m sample_factory.algorithms.appo.train_appo --env=dmlab_30 --train_for_seconds=3600000 --algo=APPO '
    '--gamma=0.99 --use_rnn=True --num_workers=90 --num_envs_per_worker=12 --ppo_epochs=1 --rollout=32 '
    '--recurrence=32 --batch_size=2048 --benchmark=False --ppo_epochs=1 --max_grad_norm=0.0 --dmlab_renderer=software '
    '--decorrelate_experience_max_seconds=120 --reset_timeout_seconds=300 --encoder_custom=dmlab_instructions '
    '--encoder_type=resnet --encoder_subtype=resnet_impala --encoder_extra_fc_layers=1 --hidden_size=256 '
    '--nonlinearity=relu --rnn_type=lstm --dmlab_extended_action_set=True --num_policies=1 '
    '--with_pbt=False --experiment=sf_ori_dmlab_30 '
    '--dmlab_one_task_per_worker=True --set_workers_cpu_affinity=True --max_policy_lag=35 '
    '--pbt_target_objective=dmlab_target_objective '
    '--dmlab30_dataset=/home/zhehui/mixppo/dmlab/datasets/brady_konkle_oliva2008 '
    '--dmlab_use_level_cache=True --dmlab_level_cache_path=/home/zhehui/mixppo/dmlab/.dmlab_cache'
)