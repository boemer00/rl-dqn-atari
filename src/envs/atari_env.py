from collections import deque
import gym
import numpy as np
from cv2

class AtariEnv:
    def __init__(self, env_name, frame_skip=4, frame_stack=4):
        self.env = gym.make(env_name)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def reset(self):
        state = self.env.reset()
        state = self.preprocess(state)
        for _ in range(self.frame_stack):
            self.frames.append(state)
        return np.stack(self.frames, axis=0)

    def step(self, action):
        total_reward = 0
        done = False
        for _ in range(self.frame_skip):
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        next_state = self.preprocess(next_state)
        self.frames.append(next_state)
        return np.stack(self.frames, axis=0), total_reward, done, info

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = frame / 255.0
        return frame

    def render(self):
        self.env.render()
