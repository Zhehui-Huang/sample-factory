python -m sample_factory.runner.run \
--run=runs.dmlab.dmlab30 \
--runner=slurm --slurm_workdir=sf2_slurm_output \
--experiment_suffix=slurm --pause_between=1 \
--slurm_gpus_per_job=1 --slurm_cpus_per_gpu=72 \
--slurm_sbatch_template=/home/zhehui/slurm/mixppo_sbatch_timeout.sh \
--slurm_print_only=False
