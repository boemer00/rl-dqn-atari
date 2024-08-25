import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class AtariEnv:
    def __init__(self, env_name, frame_skip=4, frame_stack=4):
        self.env = gym.make(env_name, render_mode="rgb_array")
        print("Environment Metadata:", self.env.metadata)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def reset(self):
        state, _ = self.env.reset()
        state = self.preprocess(state)
        for _ in range(self.frame_stack):
            self.frames.append(state)
        return np.stack(self.frames, axis=0)

    def step(self, action):
        total_reward = 0
        done = False
        for _ in range(self.frame_skip):
            next_state, reward, done, _, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        next_state = self.preprocess(next_state)
        self.frames.append(next_state)
        return np.stack(self.frames, axis=0), total_reward, done, info

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame

    def render(self):
        self.env.render(mode='rgb_array')
