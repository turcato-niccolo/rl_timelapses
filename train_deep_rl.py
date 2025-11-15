import sys
import os
import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from setproctitle import setproctitle

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter

import importlib


# ----------------------------
# Dynamic import helper
# ----------------------------
def get_class_from_module(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# ----------------------------
# Policy evaluation
# ----------------------------
def eval_policy(policy, eval_env, seed, step, max_episode_steps,
                eval_episodes=10, return_trajectory=False, writer=None):
    avg_reward = 0.
    states, actions, rewards = [], [], []

    for ep in range(eval_episodes):
        episode_states, episode_actions, episode_rewards = [], [], []
        state, done = eval_env.reset(seed=seed + ep)[0], False
        cnt = 0
        done = truncated = False
        while not (done or truncated) and cnt < max_episode_steps:
            # print(state.shape)
            action = policy.select_action(np.array(state), evaluate=True)
            next_state, reward, done, truncated, _ = eval_env.step(action)
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            avg_reward += reward
            state = next_state
            cnt += 1

        states.append(episode_states)
        actions.append(episode_actions)
        rewards.append(episode_rewards)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Step {step} - Eval over {eval_episodes} eps: {avg_reward:.3f}")
    print("---------------------------------------")

    if writer is not None:
        writer.add_scalar("eval/avg_return", avg_reward, step)

    if return_trajectory:
        return np.array(states), np.array(actions), np.array(rewards), avg_reward

    return avg_reward


# ----------------------------
# Environment builder
# ----------------------------
def make_eval_env(env_id, seed, video_dir=None, save_video=False):
    render_mode = "rgb_array" if save_video else None
    if "LunarLander" in args.env:
        env = gym.make(env_id, continuous=True, render_mode=render_mode)
    else:
        env = gym.make(env_id, render_mode=render_mode)

    if save_video and video_dir is not None:
        # Track eval counter
        eval_counter = {"count": 0}

        def episode_trigger(ep_id):
            # Increment eval_counter at each new eval call
            eval_counter["count"] += 1
            return True  # record all episodes

        # Wrapper that modifies file names dynamically
        class CustomRecordVideo(RecordVideo):
            def _get_file_path(self, episode_id):
                filename = f"eval-seed{seed}-eval{eval_counter['count']}-episode-{episode_id}.mp4"
                return os.path.join(self.video_folder, filename)

        env = CustomRecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep_id: True,
        )

    return env


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="SAC", type=str,
                        help="RL algorithm class name in deep_rl.py (e.g. SAC, TD3, DDPG)")
    parser.add_argument("--env", default="Pendulum-v1", type=str,
                        help="Gymnasium env id")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=1000, type=int)
    parser.add_argument("--max_episode_steps", default=200, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--max_timesteps", default=1000000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--buffer_size", default=1000000, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--real_update_freq", action="store_true")
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--visual_policy_input", action="store_true")

    args = parser.parse_args()

    # Base folder per algorithm
    algo_folder = os.path.join("results_tmp", args.algo if args.name is None else args.name)
    os.makedirs(algo_folder, exist_ok=True)

    # Subfolders for logs & videos
    log_dir = os.path.join(algo_folder, "runs")

    file_name = f"{args.algo}_{args.env}_{args.seed}"
    log_name = f"{args.env}_{args.seed}"

    setproctitle(f"{args.algo}||{args.env}||{algo_folder}||{args.seed}")

    video_dir = os.path.join(log_dir, file_name, "videos") if args.save_video else None
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    # Environments
    if "LunarLander" in args.env:
        env = gym.make(args.env, continuous=True)
    else:
        env = gym.make(args.env)
    eval_env = make_eval_env(args.env, args.seed, video_dir, args.save_video)
    eval_max_episode_steps = args.max_episode_steps
    eval_episodes = args.eval_episodes
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # Dynamically load algorithm class from deep_rl.py
    policy_cls = get_class_from_module("deep_rl", args.algo)
    if args.visual_policy_input:
        state_dim = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        print("max_action", max_action)
        print("action_dim", action_dim)
        print("state_dim", state_dim)


        policy = policy_cls(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=args.discount,
            tau=args.tau,
            policy_freq=args.policy_freq,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            rgb_input=True
        )
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        print("max_action", max_action)
        print("action_dim", action_dim)
        print("state_dim", state_dim)

        policy = policy_cls(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=args.discount,
            tau=args.tau,
            policy_freq=args.policy_freq,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    # Replay buffer (assumes deep_rl exports ReplayBuffer)
    ReplayBuffer = get_class_from_module("deep_rl", "ReplayBuffer")

    # Tensorboard logger in algo-specific folder
    writer = SummaryWriter(log_dir=os.path.join(log_dir, file_name))

    if args.load_model:
        policy.load(f"./{algo_folder}/{log_name}")
        states, actions, rewards, avg_return = eval_policy(
            policy, eval_env, args.seed, 0, eval_max_episode_steps,
            eval_episodes=eval_episodes, return_trajectory=True, writer=writer)
        for k in range(len(states)):
            fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
            axs[0].plot(states[k][:, 0])
            axs[1].plot(actions[k])
            axs[2].plot(rewards[k])
        plt.show()
    else:
        replay_buffer = ReplayBuffer(int(state_dim), int(action_dim), max_size=int(args.buffer_size))

        # Evaluate untrained policy
        evaluations = [eval_policy(policy, eval_env, args.seed, 0,
                                   eval_max_episode_steps, eval_episodes=eval_episodes,
                                   writer=writer)]

        state, done = env.reset(seed=args.seed)[0], False
        episode_reward, episode_timesteps, episode_num = 0, 0, 0

        for t in range(int(args.max_timesteps)):
            episode_timesteps += 1

            # Select action
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(state), evaluate=False)

            # Step
            next_state, reward, done, truncated, _ = env.step(action)
            if "Swimmer" in args.env: # Terminate if angles are too big
                q1 = next_state[1]
                q2 = next_state[2]
                if abs(q1) > np.pi/4 or abs(q2) > np.pi/4:
                    done = True

            done_bool = float(done or truncated) if episode_timesteps < args.max_episode_steps else 0

            # Store
            replay_buffer.add(state, action, next_state, reward, done_bool)
            state = next_state
            episode_reward += reward

            # Train
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)

            if done or truncated or episode_timesteps >= args.max_episode_steps:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} "
                      f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                writer.add_scalar("train/episode_reward", episode_reward, t)

                state, done = env.reset(seed=args.seed)[0], False
                episode_reward, episode_timesteps, episode_num = 0, 0, episode_num + 1

            # Evaluation
            if (t + 1) % args.eval_freq == 0:
                evaluations.append(eval_policy(
                    policy, eval_env, args.seed, (t + 1) // args.eval_freq,
                    eval_max_episode_steps, eval_episodes=eval_episodes, writer=writer))
                np.save(f"{algo_folder}/{log_name}", evaluations)
                if args.save_model:
                    policy.save(f"./{algo_folder}/{log_name}")

    writer.close()
