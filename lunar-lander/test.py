import torch
import numpy as np
import gym
import warnings
warnings.filterwarnings("ignore")

def test_agent(agent_path, env):
    lengths = []
    rewards = []
    for _ in range(10):
        done = False
        ob = env.reset()
        agent = torch.load(agent_path)
        length = 0
        reward = 0

        while not done:
            env.render()
            qs = agent(torch.from_numpy(ob).float())
            a = qs.argmax().numpy()

            next_ob, r, done, _ = env.step(a)
            ob = next_ob
            length += 1
            reward += r

        env.close()
        lengths.append(length)
        rewards.append(reward)

    print(f'average episode length: {np.mean(lengths)}')
    print(f'average reward incurred: {np.mean(rewards)}')

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent_path = './trained_agent_cartpole-v0.pt'
    test_agent(agent_path, env)

    #env = gym.make('LunarLander-v2')
    #agent_path = './trained_agent_lunarlander-v2.pt'
    #test_agent(agent_path, env)
