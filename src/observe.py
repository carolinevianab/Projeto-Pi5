import os
from nes_py.wrappers import JoypadSpace
import gym
from Contra.actions import SIMPLE_MOVEMENT
from contra import DQNAgent
from wrappers import wrapper

env = gym.make('Contra-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = wrapper(env)

# Parameters
states = (84, 84, 4)
actions = env.action_space.n

# Agent
agent = DQNAgent(states=states, actions=actions, max_memory=100000, double_q=True)

## -- main execution --- ##
os.system("cls")

print("(2021) SENAC - BCC 5° Semestre - Inteligencia Artificial - Projeto Integrador")
print("Esse projeto foi desenvolvido por: Caroline Viana, Danilo Duarte e Richard Santino\n")

n = input("Digite a quantidade de jogadas que o agente irá executar: (enter -> default = 1) ")
if(isinstance(n, str) or int(n) <= 0): n = 1

# Run the agent with a saved neural model
agent.replay(env, './models', int(n))

os.system("cls")

print("byebye :)")