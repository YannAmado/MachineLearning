import gym
env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n

env.reset()
import time
for i in range(1000):
    env.render()
    time.sleep(1)
    env.close()