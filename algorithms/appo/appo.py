import json
import math
import time
from collections import deque
from os.path import join
from queue import Empty

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.multiprocessing import Queue as TorchQueue, JoinableQueue as TorchJoinableQueue

from algorithms.appo.actor_worker import make_env_func, ActorWorker
from algorithms.appo.learner import LearnerWorker
from algorithms.appo.policy_worker import PolicyWorker
from algorithms.appo.population_based_training import PopulationBasedTraining
from envs.doom.multiplayer.doom_multiagent_wrapper import MultiAgentEnv
from utils.timing import Timing
from utils.utils import summaries_dir, experiment_dir, log, str2bool, memory_consumption_mb, cfg_file, ensure_dir_exists

torch.multiprocessing.set_sharing_strategy('file_system')


class Algorithm:
    @classmethod
    def add_cli_args(cls, parser):
        p = parser

        p.add_argument('--seed', default=None, type=int, help='Set a fixed seed value')

        p.add_argument('--initial_save_rate', default=1000, type=int, help='Save model every N train steps in the beginning of training')
        p.add_argument('--keep_checkpoints', default=2, type=int, help='Number of model checkpoints to keep')
        p.add_argument('--save_milestones_sec', default=-1, type=int, help='Save intermediate checkpoints in a separate folder for later evaluation (default=never)')

        p.add_argument('--stats_episodes', default=100, type=int, help='How many episodes to average to measure performance (avg. reward etc)')

        p.add_argument('--learning_rate', default=1e-4, type=float, help='LR')

        p.add_argument('--train_for_env_steps', default=int(1e10), type=int, help='Stop after all policies are trained for this many env steps')
        p.add_argument('--train_for_seconds', default=int(1e10), type=int, help='Stop training after this many seconds')

        # observation preprocessing
        p.add_argument('--obs_subtract_mean', default=0.0, type=float, help='Observation preprocessing, mean value to subtract from observation (e.g. 128.0 for 8-bit RGB)')
        p.add_argument('--obs_scale', default=1.0, type=float, help='Observation preprocessing, divide observation tensors by this scalar (e.g. 128.0 for 8-bit RGB)')

        # RL
        p.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
        p.add_argument(
            '--reward_scale', default=1.0, type=float,
            help=('Multiply all rewards but this factor before feeding into RL algorithm.'
                  'Sometimes the overall scale of rewards is too high which makes value estimation a harder regression task.'
                  'Loss values become too high which requires a smaller learning rate, etc.'),
        )
        p.add_argument('--reward_clip', default=10.0, type=float, help='Clip rewards between [-c, c]. Default [-10, 10] virtually means no clipping for most envs')

        # policy size and configuration
        p.add_argument('--encoder', default='convnet_simple', type=str, help='Type of the policy head (e.g. convolutional encoder)')
        p.add_argument('--hidden_size', default=512, type=int, help='Size of hidden layer in the model, or the size of RNN hidden state in recurrent model (e.g. GRU)')

    def __init__(self, cfg):
        self.cfg = cfg


class APPO(Algorithm):
    """Async PPO."""

    @classmethod
    def add_cli_args(cls, parser):
        p = parser
        super().add_cli_args(p)

        p.add_argument('--adam_eps', default=1e-6, type=float, help='Adam epsilon parameter (1e-8 to 1e-5 seem to reliably work okay, 1e-3 and up does not work)')
        p.add_argument('--adam_beta1', default=0.9, type=float, help='Adam momentum decay coefficient')
        p.add_argument('--adam_beta2', default=0.999, type=float, help='Adam second momentum decay coefficient')

        p.add_argument('--gae_lambda', default=0.95, type=float, help='Generalized Advantage Estimation discounting')

        p.add_argument('--rollout', default=64, type=int, help='Length of the rollout from each environment in timesteps. Size of the training batch is rollout X num_envs')

        p.add_argument('--num_workers', default=16, type=int, help='Number of parallel environment workers. Should be less than num_envs and should divide num_envs')

        p.add_argument('--recurrence', default=32, type=int, help='Trajectory length for backpropagation through time. If recurrence=1 there is no backpropagation through time, and experience is shuffled completely randomly')
        p.add_argument('--use_rnn', default=True, type=str2bool, help='Whether to use RNN core in a policy or not')

        p.add_argument('--ppo_clip_ratio', default=1.1, type=float, help='We use unbiased clip(x, e, 1/e) instead of clip(x, 1+e, 1-e) in the paper')
        p.add_argument('--ppo_clip_value', default=0.2, type=float, help='Maximum absolute change in value estimate until it is clipped. Sensitive to value magnitude')
        p.add_argument('--batch_size', default=1024, type=int, help='PPO minibatch size')
        p.add_argument('--ppo_epochs', default=4, type=int, help='Number of training epochs before a new batch of experience is collected')
        p.add_argument('--target_kl', default=0.02, type=float, help='Target distance from behavior policy at the end of training on each experience batch')

        p.add_argument('--normalize_advantage', default=True, type=str2bool, help='Whether to normalize advantages or not (subtract mean and divide by standard deviation)')

        p.add_argument('--max_grad_norm', default=4.0, type=float, help='Max L2 norm of the gradient vector')

        # components of the loss function
        p.add_argument(
            '--prior_loss_coeff', default=0.001, type=float,
            help=('Coefficient for the exploration component of the loss function. Typically this is entropy maximization, but here we use KL-divergence between our policy and a prior.'
                  'By default prior is a uniform distribution, and this is numerically equivalent to maximizing entropy.'
                  'Alternatively we can use custom prior distributions, e.g. to encode domain knowledge'),
        )
        p.add_argument('--initial_kl_coeff', default=0.0001, type=float, help='Initial value of KL-penalty coefficient. This is adjusted during the training such that policy change stays close to target_kl')
        p.add_argument('--kl_coeff_large', default=0.0, type=float, help='Loss coefficient for the quadratic KL term')
        p.add_argument('--value_loss_coeff', default=0.5, type=float, help='Coefficient for the critic loss')

        # APPO-specific
        p.add_argument('--num_envs_per_worker', default=2, type=int, help='Number of envs on a single CPU actor')
        p.add_argument('--worker_num_splits', default=2, type=int, help='Typically we split a vector of envs into two parts for "double buffered" experience collection')
        p.add_argument('--num_policies', default=1, type=int, help='Number of policies to train jointly')
        p.add_argument('--policy_workers_per_policy', default=1, type=int, help='Number of GPU workers that compute policy forward pass (per policy)')
        p.add_argument('--macro_batch', default=6144, type=int, help='Amount of experience to collect per policy before passing experience to the learner')
        p.add_argument('--max_policy_lag', default=25, type=int, help='Max policy lag in policy versions. Discard all experience that is older than this.')

        p.add_argument('--sync_mode', default=False, type=str2bool, help='Fully synchronous mode to compare against the standard PPO implementation')

        p.add_argument('--with_vtrace', default=True, type=str2bool, help='Enables V-trace off-policy correction')

        p.add_argument(
            '--worker_init_delay', default=0.1, type=float,
            help=('With some envs, especially multi-player Doom envs, it helps to add a delay when creating workers. It prevents too many envs from being initialized at the same time,'
                  'and reduces chance of crashes during startup'),
        )
        p.add_argument('--init_workers_parallel', default=5, type=int, help='Limit the maximum amount of workers we initialize in parallel. Helps to avoid crashes with some envs')

        # PBT stuff
        p.add_argument('--with_pbt', default=True, type=str2bool, help='Enables population-based training basic features')
        p.add_argument('--pbt_period_env_steps', default=int(8e6), type=int, help='Periodically replace the worst policies with the best ones and perturb the hyperparameters')
        p.add_argument('--pbt_replace_fraction', default=0.3, type=float, help='A portion of policies performing worst to be replace by better policies (rounded up)')
        p.add_argument('--pbt_mutation_rate', default=0.15, type=float, help='Probability that a parameter mutates')
        p.add_argument('--pbt_replace_reward_gap', default=0.1, type=float, help='Relative gap in true reward when replacing weights of the policy with a better performing one')
        p.add_argument('--pbt_replace_reward_gap_absolute', default=1e-6, type=float, help='Absolute gap in true reward when replacing weights of the policy with a better performing one')

        # debugging options
        p.add_argument('--benchmark', default=False, type=str2bool, help='Benchmark mode')
        p.add_argument('--sampler_only', default=False, type=str2bool, help='Do not send experience to the learner, measuring sampling throughput')

    def __init__(self, cfg):
        super().__init__(cfg)

        tmp_env = make_env_func(self.cfg, env_config=None)
        self.obs_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space

        self.reward_shaping_scheme = None
        if self.cfg.with_pbt:
            if hasattr(tmp_env.unwrapped, '_reward_shaping_wrapper'):
                # noinspection PyProtectedMember
                self.reward_shaping_scheme = tmp_env.unwrapped._reward_shaping_wrapper.reward_shaping_scheme
            elif isinstance(tmp_env.unwrapped, MultiAgentEnv):
                self.reward_shaping_scheme = tmp_env.unwrapped.default_reward_shaping

        tmp_env.close()

        self.actor_workers = None

        self.report_queue = TorchQueue()
        self.policy_workers = dict()
        self.policy_queues = dict()

        self.learner_workers = dict()

        self.workers_by_handle = None

        self.trajectories = dict()
        self.currently_training = set()

        self.policy_inputs = [[] for _ in range(self.cfg.num_policies)]
        self.policy_outputs = dict()
        for worker_idx in range(self.cfg.num_workers):
            for split_idx in range(self.cfg.worker_num_splits):
                self.policy_outputs[(worker_idx, split_idx)] = dict()

        self.policy_avg_stats = dict()

        self.last_timing = dict()
        self.env_steps = dict()
        self.samples_collected = [0 for _ in range(self.cfg.num_policies)]
        self.total_env_steps_since_resume = 0

        self.total_train_seconds = 0  # TODO: save and load from checkpoint??

        self.last_report = time.time()
        self.report_interval = 5.0  # sec

        self.fps_stats = deque([], maxlen=5)
        self.throughput_stats = [deque([], maxlen=5) for _ in range(self.cfg.num_policies)]
        self.avg_stats = dict()
        self.stats = dict()  # regular (non-averaged) stats

        self.writers = dict()
        writer_keys = list(range(self.cfg.num_policies))
        for key in writer_keys:
            summary_dir = join(summaries_dir(experiment_dir(cfg=self.cfg)), str(key))
            summary_dir = ensure_dir_exists(summary_dir)
            self.writers[key] = SummaryWriter(summary_dir, flush_secs=20)

        self.pbt = PopulationBasedTraining(self.cfg, self.reward_shaping_scheme, self.writers)

    def _cfg_dict(self):
        if isinstance(self.cfg, dict):
            return self.cfg
        else:
            return vars(self.cfg)

    def _save_cfg(self):
        cfg_dict = self._cfg_dict()
        with open(cfg_file(self.cfg), 'w') as json_file:
            json.dump(cfg_dict, json_file, indent=2)

    def initialize(self):
        self._save_cfg()

    def finalize(self):
        pass

    def create_actor_worker(self, idx, actor_queue, policy_worker_queues):
        learner_queues = {p: w.task_queue for p, w in self.learner_workers.items()}

        return ActorWorker(
            self.cfg, self.obs_space, self.action_space, idx, task_queue=actor_queue,
            policy_queues=self.policy_queues, report_queue=self.report_queue, learner_queues=learner_queues,
            policy_worker_queues=policy_worker_queues,
        )

    # noinspection PyProtectedMember
    def init_subset(self, indices, actor_queues, policy_worker_queues):
        workers = dict()
        started_reset = dict()
        for i in indices:
            w = self.create_actor_worker(i, actor_queues[i], policy_worker_queues)
            w.init()
            time.sleep(self.cfg.worker_init_delay)  # just in case
            w.request_reset()
            workers[i] = w
            started_reset[i] = time.time()

        fastest_reset_time = None
        workers_finished = set()

        while len(workers_finished) < len(workers):
            for w in workers.values():
                if w.task_queue.qsize() > 0:
                    time.sleep(0.05)
                    continue

                if len(workers_finished) <= 0:
                    fastest_reset_time = time.time() - started_reset[w.worker_idx]
                    log.debug('Fastest reset in %.3f seconds', fastest_reset_time)

                if not w.critical_error.is_set():
                    workers_finished.add(w.worker_idx)

            if workers_finished:
                log.warning('Workers finished: %r', workers_finished)

            for worker_idx, w in workers.items():
                if worker_idx in workers_finished:
                    continue

                time_passed = time.time() - started_reset[w.worker_idx]
                if fastest_reset_time is None:
                    timeout = False
                else:
                    timeout = time_passed > max(fastest_reset_time * 1.5, fastest_reset_time + 10)

                is_process_alive = w.process.is_alive() and not w.critical_error.is_set()

                if timeout or not is_process_alive:
                    # if it takes more than 1.5x the usual time to reset, this worker is probably stuck
                    log.error('Worker %d seems to be stuck (%.3f). Reset!', w.worker_idx, time_passed)
                    log.debug('Status: %r %r', is_process_alive, w.critical_error.is_set())
                    stuck_worker = w
                    stuck_worker.process.kill()

                    new_worker = self.create_actor_worker(worker_idx, actor_queues[worker_idx], policy_worker_queues)
                    new_worker.init()
                    new_worker.request_reset()
                    started_reset[worker_idx] = time.time()

                    workers[worker_idx] = new_worker
                    del stuck_worker

        return workers.values()

    # noinspection PyUnresolvedReferences
    def init_workers(self):
        actor_queues = [TorchQueue() for _ in range(self.cfg.num_workers)]

        policy_worker_queues = dict()
        for policy_id in range(self.cfg.num_policies):
            policy_worker_queues[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                policy_worker_queues[policy_id].append(TorchJoinableQueue())

        log.info('Initializing GPU learners...')
        learner_idx = 0
        for policy_id in range(self.cfg.num_policies):
            learner_worker = LearnerWorker(
                learner_idx, policy_id, self.cfg, self.obs_space, self.action_space,
                self.report_queue, policy_worker_queues[policy_id],
            )
            learner_worker.start_process()
            learner_worker.init()

            self.learner_workers[policy_id] = learner_worker
            learner_idx += 1

        log.info('Initializing GPU workers...')
        for policy_id in range(self.cfg.num_policies):
            self.policy_workers[policy_id] = []

            policy_queue = TorchQueue()
            self.policy_queues[policy_id] = policy_queue

            for i in range(self.cfg.policy_workers_per_policy):
                policy_worker = PolicyWorker(
                    i, policy_id, self.cfg, self.obs_space, self.action_space,
                    policy_queue, actor_queues, self.report_queue, policy_worker_queues,
                )
                self.policy_workers[policy_id].append(policy_worker)
                policy_worker.start_process()

        log.info('Initializing actors...')

        self.actor_workers = []
        max_parallel_init = self.cfg.init_workers_parallel
        worker_indices = list(range(self.cfg.num_workers))
        for i in range(0, self.cfg.num_workers, max_parallel_init):
            workers = self.init_subset(worker_indices[i:i + max_parallel_init], actor_queues, policy_worker_queues)
            self.actor_workers.extend(workers)

    def init_pbt(self):
        if self.cfg.with_pbt:
            self.pbt.init(self.learner_workers, self.actor_workers)

    def finish_initialization(self):
        """Wait until policy workers are fully initialized."""
        for policy_id, workers in self.policy_workers.items():
            for w in workers:
                log.debug('Waiting for policy worker %d-%d to finish initialization...', policy_id, w.worker_idx)
                w.init()
                log.debug('Policy worker %d-%d initialized!', policy_id, w.worker_idx)

    def process_report(self, report):
        if 'policy_id' in report:
            policy_id = report['policy_id']

            if 'env_steps' in report:
                if policy_id in self.env_steps:
                    delta = report['env_steps'] - self.env_steps[policy_id]
                    self.total_env_steps_since_resume += delta
                self.env_steps[policy_id] = report['env_steps']

            if 'episodic' in report:
                s = report['episodic']
                for key, value in s.items():
                    if key not in self.policy_avg_stats:
                        self.policy_avg_stats[key] = [deque(maxlen=100) for _ in range(self.cfg.num_policies)]

                    self.policy_avg_stats[key][policy_id].append(value)

            if 'train' in report:
                self.report_train_summaries(report['train'], policy_id)

            if 'samples' in report:
                self.samples_collected[policy_id] += report['samples']

        if 'timing' in report:
            for k, v in report['timing'].items():
                if k not in self.avg_stats:
                    self.avg_stats[k] = deque([], maxlen=50)
                self.avg_stats[k].append(v)

        if 'stats' in report:
            self.stats.update(report['stats'])

    def report(self):
        now = time.time()

        total_env_steps = sum(self.env_steps.values())
        self.fps_stats.append((now, total_env_steps))
        if len(self.fps_stats) <= 1:
            return

        past_moment, past_frames = self.fps_stats[0]
        fps = (total_env_steps - past_frames) / (now - past_moment)

        sample_throughput = dict()
        for policy_id in range(self.cfg.num_policies):
            self.throughput_stats[policy_id].append((now, self.samples_collected[policy_id]))
            if len(self.throughput_stats[policy_id]) > 1:
                past_moment, past_samples = self.throughput_stats[policy_id][0]
                sample_throughput[policy_id] = (self.samples_collected[policy_id] - past_samples) / (now - past_moment)
            else:
                sample_throughput[policy_id] = math.nan

        self.print_stats(fps, sample_throughput, total_env_steps)
        self.report_basic_summaries(fps, sample_throughput)

    def print_stats(self, fps, sample_throughput, total_env_steps):
        samples_per_policy = ', '.join([f'{p}: {s:.1f}' for p, s in sample_throughput.items()])
        log.debug(
            'Fps is %.1f. Total num frames: %d. Throughput: %s. Samples: %d',
            fps, total_env_steps, samples_per_policy, sum(self.samples_collected),
        )

        if 'reward' in self.policy_avg_stats:
            policy_reward_stats = []
            for policy_id in range(self.cfg.num_policies):
                reward_stats = self.policy_avg_stats['reward'][policy_id]
                if len(reward_stats) > 0:
                    policy_reward_stats.append((policy_id, f'{np.mean(reward_stats):.3f}'))
            log.debug('Avg episode reward: %r', policy_reward_stats)

    def report_train_summaries(self, stats, policy_id):
        for key, scalar in stats.items():
            self.writers[policy_id].add_scalar(f'train/{key}', scalar, self.env_steps[policy_id])

    def report_basic_summaries(self, fps, sample_throughput):
        memory_mb = memory_consumption_mb()

        default_policy = 0
        for policy_id, env_steps in self.env_steps.items():
            if policy_id == default_policy:
                self.writers[policy_id].add_scalar('0_aux/fps', fps, env_steps)
                self.writers[policy_id].add_scalar('0_aux/master_process_memory_mb', float(memory_mb), env_steps)
                for key, value in self.avg_stats.items():
                    if len(value) >= value.maxlen:
                        self.writers[policy_id].add_scalar(f'stats/{key}', np.mean(value), env_steps)

                for key, value in self.stats.items():
                    self.writers[policy_id].add_scalar(f'stats/{key}', value, env_steps)

            if not math.isnan(sample_throughput[policy_id]):
                self.writers[policy_id].add_scalar('0_aux/sample_throughput', sample_throughput[policy_id], env_steps)

            for key, stat in self.policy_avg_stats.items():
                if len(stat[policy_id]) > 0:
                    stat_value = np.mean(stat[policy_id])
                    self.writers[policy_id].add_scalar(f'0_aux/avg_{key}', float(stat_value), env_steps)

    def _should_end_training(self):
        end = len(self.env_steps) > 0 and all(s > self.cfg.train_for_env_steps for s in self.env_steps.values())
        end |= self.total_train_seconds > self.cfg.train_for_seconds

        if self.cfg.benchmark:
            end |= self.total_env_steps_since_resume >= int(2e6)
            end |= sum(self.samples_collected) >= int(5e5)

        return end

    def learn(self):
        self.init_workers()
        self.init_pbt()
        self.finish_initialization()

        log.info('Collecting experience...')

        timing = Timing()
        with timing.timeit('experience'):
            try:
                while not self._should_end_training():
                    while True:
                        try:
                            report = self.report_queue.get(timeout=0.001)
                            self.process_report(report)
                        except Empty:
                            break

                    if time.time() - self.last_report > self.report_interval:
                        self.report()

                        now = time.time()
                        self.total_train_seconds += now - self.last_report
                        self.last_report = now

                    self.pbt.update(self.env_steps, self.policy_avg_stats)
                    time.sleep(0.1)

            except Exception:
                log.exception('Exception in driver loop')
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected in driver loop, exiting...')

        all_workers = self.actor_workers
        for workers in self.policy_workers.values():
            all_workers.extend(workers)
        all_workers.extend(self.learner_workers.values())

        time.sleep(1.0)
        for i, w in enumerate(all_workers):
            log.debug('Closing worker #%d...', i)
            w.close()
            time.sleep(0.01)
        for i, w in enumerate(all_workers):
            w.join()
            log.debug('Worker #%d joined!', i)

        fps = sum(self.env_steps.values()) / timing.experience
        log.info('Collected %r, FPS: %.1f', self.env_steps, fps)
        log.info('Timing: %s', timing)

        time.sleep(0.1)
        log.info('Done!')
