# Deep Convolutional Neural Network - CNN
# Reinforcemente Learning - Trial and error

from environment import Environment
from train import Trainer
from dqn import DQN

# initialize gym environment and dqn
env = Environment(args)
agent = DQN(env, args)

# train agent
Trainer(agent).run()

# play the game
env.gym.monitor.start(args.out, force=True)
agent.play()

env.gym.monitor.close()