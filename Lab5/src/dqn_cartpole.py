# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import ale_py
import os
import cv2
from collections import deque
import wandb
import argparse
import torch.nn.functional as F

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
           nn.Linear(input_dim, 64),
           nn.ReLU(),
           nn.Linear(64, 64),
           nn.ReLU(),
           nn.Linear(64, num_actions)
        )       

    def forward(self, x):
        return self.network(x)
 
class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        """新增一筆 transition 並以 error 設定其 priority"""
        # 計算 priority = (|TD-error| + eps) ** alpha
        priority = (abs(error) + self.eps) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        # 環狀覆蓋
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """依 priorities 抽樣，並回傳 IS weights 與 indices 以供後續更新"""
        # 目前有效的 buffer 長度
        buffer_len = len(self.buffer)
        # 計算抽樣機率
        probs = self.priorities[:buffer_len]
        probs = probs / probs.sum()

        # 按機率抽 indices
        indices = np.random.choice(buffer_len, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # 計算 importance-sampling weights
        # w_i = (N * P(i))^(-beta), 並正規化使 max(w)=1
        N = buffer_len
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # 解包 transitions
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32)
        )

    def update_priorities(self, indices, errors):
        """根據新的 TD-error 更新對應 samples 的 priority"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + self.eps) ** self.alpha
    
    def __len__(self):
        return len(self.buffer)
  
        
class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        self.num_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(4, self.num_actions).to(self.device)  # CartPole has 4 state dimensions
        self.q_net.apply(init_weights)
        self.target_net = DQN(4, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.memory = PrioritizedReplayBuffer(args.memory_size)
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = obs
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = next_obs
                self.memory.add((state, action, reward, next_state, done), 0)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = obs  # No preprocessing for CartPole
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_obs

        return total_reward



    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 

        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, indices, weights = batch

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

       # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = (weights * F.mse_loss(q_values, target_q, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priorities(indices, (q_values - target_q).detach().abs().cpu().numpy())

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results/dqn_cartpole")
    parser.add_argument("--wandb-run-name", type=str, default="dqn_cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()