import os
import torch
import time
import imageio
import argparse
from tqdm import tqdm
from src.envs.atari_env import AtariEnv
from src.agents.dqn import DQNAgent

def get_device():
    return torch.device("cpu")  # Force CPU usage for MacBook Pro 2013

def evaluate(env_name, model_path, num_episodes=1, save_video=True, frame_save_freq=1):
    device = get_device()
    print(f"Using device: {device}")

    env = AtariEnv(env_name)
    state_dim = (4, 84, 84)
    action_dim = env.env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    try:
        agent.q_network.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    agent.q_network.to(device)
    agent.epsilon = 0  # No exploration during evaluation

    all_frames = []
    total_rewards = []

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        state = env.reset()
        total_reward = 0
        done = False
        episode_frames = []

        while not done:
            start_time = time.time()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.select_action(state_tensor)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

            if save_video and episode % frame_save_freq == 0:
                frame = env.env.render()
                episode_frames.append(frame)

            elapsed_time = time.time() - start_time
            time.sleep(max(0, 1/30 - elapsed_time))  # Assuming 30 FPS

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        if save_video and episode % frame_save_freq == 0:
            all_frames.extend(episode_frames)

    print(f"Average Total Reward: {sum(total_rewards) / len(total_rewards)}")

    if save_video and all_frames:
        video_path = 'breakout_eval.mp4'
        imageio.mimsave(video_path, all_frames, fps=30)
        print(f"Evaluation video saved to {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN agent on Atari Breakout")
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5", help="Atari environment name")
    parser.add_argument("--model", type=str, default="atari_dqn.pth", help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to evaluate")
    parser.add_argument("--no-video", action="store_true", help="Disable video saving")
    parser.add_argument("--frame-freq", type=int, default=1, help="Save every nth frame")
    args = parser.parse_args()

    evaluate(args.env, args.model, args.episodes, not args.no_video, args.frame_freq)
