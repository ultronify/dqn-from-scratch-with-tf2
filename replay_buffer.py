import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Replay Buffer

    Stores and retrieves gameplay experiences
    """

    def __init__(self):
        self.gameplay_experiences = deque(maxlen=1000000)

    def store_gameplay_experience(self, state, next_state, reward, action,
                                  done):
        """
        Records a single step (state transition) of gameplay experience.

        :param state: the current game state
        :param next_state: the game state after taking action
        :param reward: the reward taking action at the current state brings
        :param action: the action taken at the current state
        :param done: a boolean indicating if the game is finished after
        taking the action
        :return: None
        """
        self.gameplay_experiences.append((state, next_state, reward, action,
                                          done))

    def sample_gameplay_batch(self):
        """
        Samples a batch of gameplay experiences for training.

        :return: a list of gameplay experiences
        """
        batch_size = min(128, len(self.gameplay_experiences))
        sampled_gameplay_batch = random.sample(
            self.gameplay_experiences, batch_size)
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for gameplay_experience in sampled_gameplay_batch:
            state_batch.append(gameplay_experience[0])
            next_state_batch.append(gameplay_experience[1])
            reward_batch.append(gameplay_experience[2])
            action_batch.append(gameplay_experience[3])
            done_batch.append(gameplay_experience[4])
        return np.array(state_batch), np.array(next_state_batch), np.array(
            action_batch), np.array(reward_batch), np.array(done_batch)
