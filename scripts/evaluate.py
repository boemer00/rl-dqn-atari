import torch
from src.envs.atari_env import AtariEnv
from src.agents.dqn import DQNAgent

def evaluate():
    env_name = "Breakout-v0"
    env = AtariEnv(env_name)
    state_dim = (4, 84, 84)
    action_dim = env.env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.q_network.load_state_dict(torch.load("breakout_dqn.pth"))
    agent.epsilon = 0  # No exploration during evaluation

    num_episodes = 10
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            env.render()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    evaluate()
