import torch
from torch import optim
from torch.nn import functional as F
from network import QNetwork
import gym
from tqdm import tqdm
from buffer import ReplayBuffer
from itertools import count
import argparse
import numpy as np
import utils
import warnings
warnings.filterwarnings("ignore")

print(torch.__version__)

def experiment(args):
    # environment setup
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n # here it is discrete, so we have n here as opposed to the dimension of the action

    # network setup
    network = QNetwork(args.gamma, state_dim, action_dim, args.hidden_sizes)

    # optimizer setup
    if args.env == 'CartPole-v0':
        optimizer = optim.RMSprop(network.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(network.parameters(), lr=args.lr)

    # target setup (if wanted)
    if args.target:
        target_network = QNetwork(args.gamma, state_dim, action_dim, args.hidden_sizes)
        target_network.load_state_dict(network.state_dict())
        target_network.eval()

    # buffer setup
    buffer = ReplayBuffer(maxsize=args.max_size)

    # training
    for i in tqdm(range(args.num_episodes)):
        # initial observation, cast into a torch tensor
        ob = torch.from_numpy(env.reset()).float()

        for t in count():
            with torch.no_grad():
                # Collect the action from the policy.
                eps = utils.get_eps(args.eps, i)
                action = network.get_action(ob, eps)

            # Step the environment, convert everything to torch tensors
            n_ob, rew, done, _ = env.step(action)

            action = torch.tensor(action)
            n_ob = torch.from_numpy(n_ob).float()
            rew = torch.tensor([rew])

            # Add new experience to replay buffer.
            buffer.add_experience(ob, action, rew, n_ob, done)

            ob = n_ob

            if len(buffer) >= args.batch_size:
                if t % args.learning_freq == 0:
                    # Sample batch from replay buffer and optimize model via gradient descent.
                    states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)

                    # Compute Q values for current nd target networks
                    q = network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    if args.target:
                        target_q = target_network.get_targets(rewards, next_states, dones)
                    else:
                        target_q = network.get_targets(rewards, next_states, dones)

                    # Gradient Descent
                    optimizer.zero_grad()
                    loss = F.mse_loss(q, target_q)
                    loss.backward()
                    optimizer.step()

            if done: break

        # Update target based on args.target_update_freq.
        if args.target and i % args.target_update_freq == 0:
            utils.update_target(network, target_network, args.ema_param)

    # save final agent
    save_path = args.save_path + '_' + args.env.lower() + '.pt'
    torch.save(network, save_path)

def get_args():
    parser = argparse.ArgumentParser(description='Q-Learning')

    # Environment args
    parser.add_argument('--env', default='CartPole-v0', help='name of environment')
    parser.add_argument('--gamma', type=float, default=0.999, help='discount factor')
    parser.add_argument('--eps', type=float, default=0.999, help='epsilon parameter')

    # Network args
    parser.add_argument('--hidden_sizes', nargs='+', type=int, help='hidden sizes of Q network')
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate for Q function optimizer')
    parser.add_argument('--target', action='store_true', help='if we want to use a target network')
    parser.add_argument('--target_update_freq', type=int, default=10, help='how often we update the target network')
    parser.add_argument('--ema_param', type=float, default=1.0, help='target update parameter')

    # Replay buffer args
    parser.add_argument('--max_size', type=int, default=10000, help='max buffer size')

    # Training/saving args
    parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes to run during training')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--save_path', default='./trained_agent', help='agent save path')
    parser.add_argument('--learning_freq', type=int, default=1, help='how often to update the network after collecting experience')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    args = get_args()
    experiment(args)
