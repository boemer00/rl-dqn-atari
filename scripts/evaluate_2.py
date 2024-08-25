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

def evaluate(env_name, model_path, num_episodes=1, save_video=True, frame_save_freq=1, max_steps=10000):
    device = get_device()
    print(f"Using device: {device}")

    print("Initializing environment...")
    env = AtariEnv(env_name)
    state_dim = (4, 84, 84)
    action_dim = env.env.action_space.n
    print(f"Environment initialized. Action space: {action_dim}")

    print("Creating agent...")
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    print("Agent created.")
    
    try:
        print(f"Loading model from {model_path}...")
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
        print(f"\nStarting episode {episode + 1}")
        state = env.reset()
        print("Environment reset")
        total_reward = 0
        done = False
        episode_frames = []
        step = 0

        while not done and step < max_steps:
            print(f"Step {step}")
            print(f"State shape: {state.shape}, Type: {state.dtype}, Min: {np.min(state)}, Max: {np.max(state)}")
            start_time = time.time()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            action = q_values.argmax().item()
            print(f"Q-values: {q_values.numpy()}")
            print(f"Selected action: {action}")
            next_state, reward, done, _ = env.step(action)
            print(f"Reward: {reward}, Done: {done}")
            state = next_state
            total_reward += reward

            if save_video and episode % frame_save_freq == 0:
                frame = env.env.render()
                episode_frames.append(frame)

            elapsed_time = time.time() - start_time
            time.sleep(max(0, 1/30 - elapsed_time))  # Assuming 30 FPS
            step += 1

        if step >= max_steps:
            print(f"Episode terminated after reaching maximum steps: {max_steps}")

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1} completed, Total Reward: {total_reward}")

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
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum number of steps per episode")
    args = parser.parse_args()

    evaluate(args.env, args.model, args.episodes, not args.no_video, args.frame_freq, args.max_steps)
