import torch
import time
import imageio
from src.envs.atari_env import AtariEnv
from src.agents.dqn import DQNAgent

def evaluate():
    env_name = "ALE/Breakout-v5"
    env = AtariEnv(env_name)
    state_dim = (4, 84, 84)
    action_dim = env.env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.q_network.load_state_dict(torch.load("atari_dqn.pth"))
    agent.epsilon = 0  # No exploration during evaluation

    frames = []
    num_episodes = 1
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            start_time = time.time()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

            # Capture frame
            frame = env.env.render()
            frames.append(frame)

            # Control the frame rate manually if needed
            elapsed_time = time.time() - start_time
            time.sleep(max(0, 1/30 - elapsed_time))  # Assuming 30 FPS

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Save frames as a video
    imageio.mimsave('breakout_eval.mp4', frames, fps=30)

if __name__ == "__main__":
    evaluate()
