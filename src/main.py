import torch
from agents.dqn import DQNAgent
from envs.atari_env import AtariEnv

def main():
    env_name = "Breakout-v0"
    env = AtariEnv(env_name)
    state_dim = (4, 84, 84)
    action_dim = env.env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    num_episodes = 10
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        agent.update_target_network()

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    # Save the model
    torch.save(agent.q_network.state_dict(), "breakout_dqn.pth")

if __name__ == "__main__":
    main()
