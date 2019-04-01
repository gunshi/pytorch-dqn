import gym
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule
from utils.seed import set_global_seeds
from gym import wrappers
from utils.atari_wrapper import *

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 100000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

def main(env, num_timesteps):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

def get_env2(env_name, seed):

    env = gym.make(env_name)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = 'tmp/gym-results'
    env = wrappers.Monitor(env, expt_dir, force=True)
    # env = wrap_deepmind(env)

    return env

if __name__ == '__main__':
    # Get Atari games.
    # benchmark = gym.benchmark_spec('Atari40M')

    # # Change the index to select a different game.
    # task = benchmark.tasks[3]
    # env=gym.make('Breakout-v0')
    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env2('MsPacman-v0', seed)

    main(env, 1000000)