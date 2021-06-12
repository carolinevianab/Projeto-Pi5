from nes_py.wrappers import JoypadSpace
import gym
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from contra import DQNAgent
from wrappers import wrapper

env = gym.make('Contra-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper(env)

# Parameters
states = (84, 84, 4)
actions = env.action_space.n

# Agent
agent = DQNAgent(states=states, actions=actions, max_memory=100000, double_q=True)

# Run the agent with a saved neural model
agent.replay(env, './models', 1, False)