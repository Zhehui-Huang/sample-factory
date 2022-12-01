python -m sample_factory.launcher.run \
--backend=slurm \
--experiment_suffix=slurm \
--slurm_gpus_per_job=1 \
--slurm_cpus_per_gpu=16 \
--pause_between=1 \
--slurm_print_only=False \
--slurm_sbatch_template=/home/zhehui/reward_decrease/slurm/sbatch_timeout.sh \
--slurm_workdir=/home/zhehui/reward_decrease/slurm/mujoco \
--run=sf_examples.mujoco.experiments.mujoco_pendulum