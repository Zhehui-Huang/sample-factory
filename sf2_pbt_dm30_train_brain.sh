python -m sample_factory.launcher.run \
--run=runs.dmlab.dmlab30_pbt \
--backend=slurm --slurm_workdir=sf2_slurm_output \
--experiment_suffix=slurm --pause_between=1 \
--slurm_gpus_per_job=4 --slurm_cpus_per_gpu=18 \
--slurm_sbatch_template=/home/zhehui/slurm/mixppo_sbatch_timeout.sh \
--slurm_print_only=False