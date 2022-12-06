python -m sample_factory.runner.run \
--runner=slurm \
--experiment_suffix=slurm \
--slurm_gpus_per_job=1 \
--slurm_cpus_per_gpu=16 \
--pause_between=1 \
--slurm_print_only=False \
--slurm_sbatch_template=/home/zhehui/reward_decrease/sf_july/slurm/sbatch_timeout.sh \
--slurm_workdir=/home/zhehui/reward_decrease/sf_july/slurm/mujoco/hopper \
--run=sample_factory_examples.mujoco_examples.mujoco_hopper