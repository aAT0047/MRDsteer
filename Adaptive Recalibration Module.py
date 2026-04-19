#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
import copy
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import typing
from numpy.random import default_rng
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#argument parameters
parser = argparse.ArgumentParser(description='Allow user to pass agent hyperparameters')
parser.add_argument('--episodes', type=int, help='number of episodes to train', default=50000)
parser.add_argument('--trainsteps', type=int, help='training step interval', default=8)
parser.add_argument('--updatesteps', type=int, help='target update step interval', default=10000)
parser.add_argument('--batchsize', type=int, help='number of experiences to sample from memory', default=32)
parser.add_argument('--alpha', type=float, help='alpha to use for prioritized experience replay', default=0.3)

# =========================
# Environment
# =========================
class Environment():
    """
    State: (d_t, u_t)
    Action: 0=continue, 1=rollback
    Costs:
        continue: base_cost = d_t + kappa / ( (u_t / rul_scale) + eps )
                  + C_fail if failure boundary reached
        rollback: C_cb (fixed)
    Reward: -cost
    Rollback EFFECT (non-terminal):
        d_new = max(0, d_old - rollback_delta_d)
        u_new = min(u_max, u_old + rollback_delta_u)
        downtime_remaining = downtime_steps  (期间 continue 不推进退化/rul)
    Failure flag:
        fail if (d_t >= d_fail) or (u_t <= 0)
    """
    def __init__(self,
                 csv_path='train.csv',
                 kappa=0.5,
                 eps=1e-6,
                 C_fail=10.0,
                 C_cb=3.0,
                 d_fail=0.2,
                 rul_scale=1.00,
                 rollback_delta_d=0.4,
                 rollback_delta_u=15.0,
                 downtime_steps=3,
                 terminate_on_hard_fail=True,
                 max_episode_steps=1000):
        self.dataset = pd.read_csv(csv_path)
        self.grouped = self.dataset.groupby('sample_id')
        self.sample_ids = list(self.grouped.groups.keys())

        # Cost parameters
        self.kappa = kappa
        self.eps = eps
        self.C_fail = C_fail
        self.C_cb = C_cb
        self.d_fail = d_fail
        self.rul_scale = rul_scale

        # Rollback dynamics
        self.rollback_delta_d = rollback_delta_d
        self.rollback_delta_u = rollback_delta_u
        self.downtime_steps = downtime_steps

        self.terminate_on_hard_fail = terminate_on_hard_fail
        self.max_episode_steps = max_episode_steps

        # Internal state
        self.reset()

    def _pick_trajectory(self):
        sid = random.choice(self.sample_ids)
        df_traj = self.grouped.get_group(sid).reset_index(drop=True)
        # If you have an ordering column (e.g., 'row'), you can sort:
        # df_traj = df_traj.sort_values('row').reset_index(drop=True)
        d_seq = df_traj['X_pred'].to_numpy(dtype=np.float32)
        df_traj['rul'] += 20
        u_seq = df_traj['rul'].to_numpy(dtype=np.float32)
        return sid, d_seq, u_seq

    def reset(self):
        self.cycle = 0
        self.sid, self.d_seq, self.u_seq = self._pick_trajectory()
        self.steps_elapsed = 0
        self.downtime_remaining = 0
        # For rul restoration capping
        self.u_max = float(np.max(self.u_seq)) if len(self.u_seq) > 0 else 0.0
        return self.get_state()

    def get_state(self):
        d_t = self.d_seq[self.cycle]
        u_t = self.u_seq[self.cycle]
        return np.array([d_t, u_t], dtype=np.float32)

    def _failure_flag(self, d_t, u_t):
            # 修正: 对应材料中 x_k >= y_th 或 l_k <= 0 的灾难性失效边界
            return (d_t >= self.d_fail) or (u_t <= 0)

    def _advance_degradation(self):
        """
        Advance to next time index if possible; if at end keep last but mark termination.
        In original code, episode ended when sequence ended. We'll keep same.
        """
        self.cycle += 1
        if self.cycle >= len(self.d_seq):
            return True  # sequence exhausted
        return False

    def _apply_rollback(self, d_t, u_t):
        """
        Apply rollback effect: reduce d, restore some u, clamp to bounds.
        """
        d_new = max(0.0, d_t - self.rollback_delta_d)
        u_new = min(self.u_max, u_t + self.rollback_delta_u)
        # Replace current point values (simulate maintenance snapshot)
        self.d_seq[self.cycle] = d_new
        self.u_seq[self.cycle] = u_new
        self.downtime_remaining = self.downtime_steps

    def take_action(self, action):
            """
            Returns: (next_state or None, reward, terminated, info)
            """
            self.steps_elapsed += 1
            d_t = self.d_seq[self.cycle]
            u_t = self.u_seq[self.cycle]

            fail_flag = self._failure_flag(d_t, u_t)

            if action == 0:  # continue
                # 修正: 严格对应公式 C_k^{base} = x_k + kappa / (l_k + eps)
                base_cost = d_t + self.kappa / (u_t + self.eps)
                cost = base_cost + (self.C_fail if fail_flag else 0.0)
                reward = -cost

                terminated = False
                
                # Hard fail termination option
                if fail_flag and self.terminate_on_hard_fail:
                    terminated = True
                    next_state = None
                    return next_state, reward, terminated, {
                        "cost": cost, "fail_flag": fail_flag, "action": action
                    }

                # Downtime logic
                if self.downtime_remaining > 0:
                    # During downtime we do NOT advance cycle
                    self.downtime_remaining -= 1
                else:
                    # Advance to next point in trajectory
                    seq_end = self._advance_degradation()
                    if seq_end:
                        terminated = True
                        next_state = None
                        return next_state, reward, terminated, {
                            "cost": cost, "fail_flag": fail_flag, "action": action
                        }

                # Episode length cap
                if self.steps_elapsed >= self.max_episode_steps:
                    terminated = True
                    next_state = None
                else:
                    next_state = self.get_state() if not terminated else None

                return next_state, reward, terminated, {
                    "cost": cost, "fail_flag": fail_flag, "action": action
                }

            elif action == 1:  # rollback
                # 修正: 回滚干预成本为固定的 C_cb
                cost = self.C_cb
                reward = -cost

                # 修正: 应用系统回滚及降温(Downtime)效果，维持系统连续处理状态
                self._apply_rollback(d_t, u_t)

                terminated = False
                # 仅在达到最大步数时截断回合
                if self.steps_elapsed >= self.max_episode_steps:
                    terminated = True
                    next_state = None
                else:
                    next_state = self.get_state()

                return next_state, reward, terminated, {
                    "cost": cost, "fail_flag": fail_flag, "action": action, "rollback": True
                }
            else:
                raise ValueError("Invalid action")
# =========================
# Transition & Memory
# =========================
class Transition():
    def __init__(self, state, action, state_new, reward, term ):
        self.state = state
        self.action = action
        self.state_new = state_new
        self.reward = reward
        self.term = term

class PrioritizedReplayMemory:
    def __init__(self, batch_size: int, buffer_size: int, alpha: float = 0.0, random_state: np.random.RandomState = None) -> None:
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer_length = 0
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("transition", Transition)])
        self._alpha = alpha
        self._random_state = np.random.RandomState() if random_state is None else random_state

    def __len__(self) -> int:
        return self._buffer_length

    def add(self, transition: Transition) -> None:
        priority = 1.0 if self._buffer_length == 0 else self._buffer["priority"].max()
        if self._buffer_length < self._buffer_size:
            self._buffer[self._buffer_length] = (priority, transition)
            self._buffer_length += 1
        else:
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, transition)

    def sample(self, beta: float) -> typing.Tuple[np.array, np.array, np.array]:
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size), size=self._batch_size, replace=True, p=sampling_probs)
        transitions = self._buffer["transition"][idxs]
        weights = (self._buffer_length * sampling_probs[idxs])**-beta
        normalized_weights = weights / weights.max()
        return idxs, transitions, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        self._buffer["priority"][idxs] = priorities

# =========================
# DQN Networks
# =========================
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(2, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 2)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

def get_action(net, state, epsilon, device):
    with torch.no_grad():
        if np.random.rand() < epsilon:
            return random.choice([0,1])
        state_t = torch.tensor(state, dtype=torch.float32).to(device)
        q_values = net(state_t)
        action = torch.argmax(q_values, dim=1).item()
        return action

# =========================
# Agent
# =========================
class Agent():
    def __init__(self, episodes=5000, trainsteps=4, updatesteps=10000, batchsize=64, alpha=0.5):
        self.exp_replay_size = 100000
        self.gamma = 0.99
        self.epsilon = 0.1
        self.min_epsilon = 0.01
        self.target_update_steps = updatesteps
        self.num_episodes = episodes
        self.batch_size = batchsize
        self.train_step_count = trainsteps
        self.steps = 0
        self.lr = 1e-3
        self.eps_decay = 1e-5
        self.loss_func = nn.MSELoss()
        self.optimizer_steps = 0
        self.optimizer_events = []
        self.alpha = alpha
        self.episode_count=0
        self.device = device
        self.QNet = DQN().to(self.device)
        self.TNet = DQN().to(self.device)
        self.TNet.load_state_dict(self.QNet.state_dict())
        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr=self.lr)
        self.ER = PrioritizedReplayMemory(batch_size=self.batch_size, buffer_size=self.exp_replay_size, alpha=self.alpha)

def optimize(agent):
    agent.optimizer_steps += 1
    beta = min(0.999, 1 - np.exp(-agent.lr * agent.episode_count))

    idxs, sample_transitions, sampling_weights = agent.ER.sample(beta=beta)

    state_batch = [t.state for t in sample_transitions]
    action_batch = [t.action for t in sample_transitions]
    reward_batch = [t.reward for t in sample_transitions]
    next_state_batch = [t.state_new for t in sample_transitions]
    term_batch = [t.term for t in sample_transitions]

    state_tensor = torch.tensor(state_batch, dtype=torch.float32, device=agent.device)
    actions_tensor = torch.tensor(action_batch, dtype=torch.int64, device=agent.device).unsqueeze(1)
    rewards_tensor = torch.tensor(reward_batch, dtype=torch.float32, device=agent.device)
    sampling_w = torch.tensor(sampling_weights, dtype=torch.float32, device=agent.device)

    # 当前 Q
    q_values_all = agent.QNet(state_tensor)              # (B, 2)
    q_values = q_values_all.gather(1, actions_tensor).squeeze(1)

    # 下一个状态的 Q (Double DQN)
    non_final_mask = torch.tensor([s is not None for s in next_state_batch],
                                  dtype=torch.bool, device=agent.device)
    target_q_next = torch.zeros(agent.batch_size, dtype=torch.float32, device=agent.device)

    if non_final_mask.any():
        non_final_next_states = torch.tensor(
            [s for s in next_state_batch if s is not None],
            dtype=torch.float32,
            device=agent.device
        )
        # 在线网选动作
        next_q_online = agent.QNet(non_final_next_states)
        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
        # 目标网估值
        next_q_target_all = agent.TNet(non_final_next_states)
        next_q_target = next_q_target_all.gather(1, next_actions).squeeze(1)
        target_q_next[non_final_mask] = next_q_target

    done_mask = torch.tensor(term_batch, dtype=torch.bool, device=agent.device)
    target_values = rewards_tensor + agent.gamma * target_q_next * (~done_mask)

    deltas = q_values - target_values.detach()

    # ---- 修正点：detach 再转 numpy ----
    priorities = deltas.detach().abs().cpu().numpy().flatten()
    agent.ER.update_priorities(idxs, priorities + 1e-5)

    loss = torch.mean((deltas * sampling_w) ** 2)

    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.QNet.parameters(), 5.0)
    agent.optimizer.step()

    return loss.item()

# =========================
# Training / Evaluation
# =========================
def main(episodes, trainsteps, updatesteps, batchsize, alpha):
    agent = Agent(episodes, trainsteps, updatesteps, batchsize, alpha)
    agent.QNet.train()
    losses = []
    cummulative_rewards = []
    epsilons = []

    for episode in tqdm(range(agent.num_episodes), desc="Training"):
        env = Environment(
            csv_path='train.csv',
            kappa=0.5,
            eps=1e-6,
            C_fail=10.0,
            C_cb=3.0,
            d_fail=0.3,
            rul_scale=1.00,          # 调整这里
            rollback_delta_d=0.4,
            rollback_delta_u=15.0,
            downtime_steps=3,
            terminate_on_hard_fail=True,
            max_episode_steps=500
        )
        state = env.reset()
        cummulative_reward = 0

        while True:
            agent.episode_count += 1
            action = get_action(agent.QNet, state, agent.epsilon, agent.device)
            next_state, reward, terminated, info = env.take_action(action)
            agent.ER.add(Transition(state, action, next_state, reward, terminated))
            cummulative_reward += reward
            agent.steps += 1
            state = next_state if next_state is not None else state  # 保持引用（终止前无所谓）

            if agent.steps % agent.train_step_count == 0 and len(agent.ER) > agent.batch_size:
                loss = optimize(agent)
                losses.append(loss)
                epsilons.append(agent.epsilon)
                agent.epsilon = max(agent.min_epsilon, agent.epsilon - agent.eps_decay)

            if agent.steps % agent.target_update_steps == 0:
                agent.TNet.load_state_dict(agent.QNet.state_dict())
                agent.optimizer_events.append(len(losses))

            if terminated:
                break

        cummulative_rewards.append(cummulative_reward)

    os.makedirs('DDQNmodel', exist_ok=True)
    np.save('DDQN-PER-wRS/losses.npy', np.array(losses))
    np.save('DDQN-PER-wRS/rewards.npy', np.array(cummulative_rewards))
    np.save('DDQN-PER-wRS/epsilons.npy', np.array(epsilons))
    torch.save(agent.QNet.state_dict(), 'DDQN-PER-wRS/model.pt')




    # —— 全局字体、字号设置 —— 
    mpl.rcParams['font.family']      = 'Times New Roman'
    mpl.rcParams['font.size']        = 20
    mpl.rcParams['axes.titlesize']   = 20
    mpl.rcParams['axes.labelsize']   = 20
    mpl.rcParams['xtick.labelsize']  = 20
    mpl.rcParams['ytick.labelsize']  = 20
    mpl.rcParams['legend.fontsize']  = 20
    mpl.rcParams['figure.titlesize'] = 20

    # —— 绘图 —— 
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle('Performance Report', fontsize=24)

    # 1) Loss & Target Updates
    if len(losses) >= 100:
        rolling_loss = np.convolve(losses, np.ones(100), 'valid') / 100
        axs[0].plot(rolling_loss, label='rolling (100) MSE', linewidth=2)
    else:
        axs[0].plot(losses, label='MSE Loss', linewidth=2)
    for event in agent.optimizer_events:
        axs[0].axvline(event, linestyle='--', alpha=0.4)
    # axs[0].set_title('(a) Loss & Target Updates')
    # axs[0].set_xlabel('Training Steps')
    # axs[0].set_ylabel('MSE')
    # axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    # 2) Cumulative Reward
    # 画业务可接受最低回报基线
    baseline = -3.0
    axs[1].axhline(baseline, color='red', linestyle='--', label='acceptable baseline')
    # 绘制滚动平均回报
    if len(cummulative_rewards) >= 100:
        rolling_rewards = np.convolve(cummulative_rewards, np.ones(100), 'valid') / 100
        axs[1].plot(rolling_rewards, label='rolling (100) Reward', linewidth=2)
    else:
        axs[1].plot(cummulative_rewards, label='Cumulative Reward', linewidth=2)
    # 标注最高点
    best_i = int(np.argmax(cummulative_rewards))
    best_r = cummulative_rewards[best_i]
    axs[1].annotate(f'{best_r:.2f}@{best_i}',
                    xy=(best_i, best_r),
                    xytext=(best_i, best_r + 0.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5))
    # axs[1].set_title('(b) Cumulative Reward')
    # axs[1].set_xlabel('Episodes')
    # axs[1].set_ylabel('Reward')
    # axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    # 3) Epsilon Decay
    axs[2].plot(epsilons, label='ε (decay)', linewidth=2)
    # axs[2].set_title('(c) Epsilon Decay')
    # axs[2].set_xlabel('Training Steps')
    # axs[2].set_ylabel('Epsilon')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)

    # —— 保存 —— 
    plt.savefig('DDQN-PER-wRS/report.png', dpi=300)
    plt.close(fig)


    # ---------- Evaluation (Train subset) ----------
    agent.QNet.eval()
    df_train = pd.read_csv('train.csv')
    grouped = df_train.groupby('sample_id')
    rng = default_rng()
    sample_ids_eval = rng.choice(grouped.ngroups, size=25, replace=False) + 1  # assuming ids start at 1

    fig, axs = plt.subplots(5, 5, constrained_layout=True)
    fig.suptitle('Suggested Rollback Points (Train)', fontsize=20)
    fig.set_figheight(10)
    fig.set_figwidth(16)

    for i, sid in enumerate(sample_ids_eval):
        row, col = divmod(i, 5)
        if sid not in grouped.groups:
            axs[row, col].title.set_text(f'sample {sid} (missing)')
            continue
        df_traj = grouped.get_group(sid).reset_index(drop=True)
        d_seq = df_traj['X_pred'].to_numpy(dtype=np.float32)
        u_seq = df_traj['rul'].to_numpy(dtype=np.float32)

        rollback_points = []
        # Simulate greedy decisions (no maintenance effect here)
        for d_t, u_t in zip(d_seq, u_seq):
            state_t = torch.tensor([d_t, u_t], dtype=torch.float32).to(agent.device)
            a = torch.argmax(agent.QNet(state_t), dim=1).item()
            if a == 1:
                rollback_points.append(len(rollback_points))  # just mark occurrence
                break

        axs[row, col].plot(d_seq, label='degradation d_t')
        if rollback_points:
            # For visualization we just mark first rollback decision index (approx)
            first_rb_index = np.where(
                [torch.argmax(agent.QNet(torch.tensor([d, u], dtype=torch.float32).to(agent.device)), dim=1).item()==1
                 for d,u in zip(d_seq,u_seq)]
            )[0]
            if len(first_rb_index)>0:
                axs[row, col].axvline(first_rb_index[0], c='red', label='rollback')
        axs[row, col].title.set_text(f'sample {sid}')
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('d_t')

    plt.savefig('DDQN-PER-wRS/train_replacements.png')
    plt.close(fig)

    # ---------- Evaluation (Test) ----------
    if os.path.exists('test.csv'):
        df_test = pd.read_csv('test.csv')
        grouped_test = df_test.groupby('sample_id')
        sample_ids_test = list(grouped_test.groups.keys())[:9]

        fig, axs = plt.subplots(3, 3, constrained_layout=True)
        fig.suptitle('Suggested Rollback Points (Test)', fontsize=20)
        fig.set_figheight(10)
        fig.set_figwidth(16)

        for i, sid in enumerate(sample_ids_test):
            row, col = divmod(i, 3)
            df_traj = grouped_test.get_group(sid).reset_index(drop=True)
            d_seq = df_traj['X_pred'].to_numpy(dtype=np.float32)
            u_seq = df_traj['rul'].to_numpy(dtype=np.float32)

            first_rb = None
            for idx, (d_t, u_t) in enumerate(zip(d_seq, u_seq)):
                state_t = torch.tensor([d_t, u_t], dtype=torch.float32).to(agent.device)
                a = torch.argmax(agent.QNet(state_t), dim=1).item()
                if a == 1:
                    first_rb = idx
                    break

            axs[row, col].plot(d_seq, label='d_t')
            if first_rb is not None:
                axs[row, col].axvline(first_rb, c='red', label='rollback')
            axs[row, col].title.set_text(f'sample {sid}')
            axs[row, col].set_xlabel('time')
            axs[row, col].set_ylabel('d_t')

        plt.savefig('DDQN-PER-wRS/test_replacements.png')
        plt.close(fig)

if __name__ == '__main__':
    args = parser.parse_args()
    main(
        episodes=args.episodes,
        trainsteps=args.trainsteps,
        updatesteps=args.updatesteps,
        batchsize=args.batchsize,
        alpha=args.alpha
    )
