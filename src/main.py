import torch
import numpy as np
from tqdm import tqdm

from src.agents.dqn import DQNAgent
from src.envs.atari_env import AtariEnv

def main():
    """
    Train a DQN agent on the Atari Breakout environment.
    """
    # Hyperparameters
    env_name = "ALE/Breakout-v5"
    num_episodes = 10000
    max_steps_per_episode = 10000
    batch_size = 32
    learning_rate = 0.00025
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.99995
    target_update_frequency = 1000
    replay_buffer_size = 100000
    min_replay_size = 10000
    save_frequency = 100

    # Initialize environment and agent
    env = AtariEnv(env_name)
    state_dim = (4, 84, 84)
    action_dim = env.env.action_space.n

    device = torch.device("cpu")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=learning_rate,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_end,
        buffer_capacity=replay_buffer_size,
        batch_size=batch_size,
        device=device
    )

    # Training loop
    total_steps = 0
    best_reward = float('-inf')

    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            total_steps += 1

            if agent.replay_buffer.size() >= min_replay_size:
                agent.train()

            if total_steps % target_update_frequency == 0:
                agent.update_target_network()

            if done:
                break

        # Epsilon decay
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}, Total Steps: {total_steps}")

        # Save the best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.q_network.state_dict(), "best_atari_dqn.pth")

        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'best_reward': best_reward,
                'epsilon': agent.epsilon
            }, f"checkpoint_episode_{episode + 1}.pth")

    # Save the final model
    torch.save(agent.q_network.state_dict(), "final_atari_dqn.pth")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Provides detailed error information
