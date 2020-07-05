"""
Training loop

This module trains the DQN agent by trial and error. In this module the DQN
agent will play the game episode by episode, store the gameplay experiences
and then use the saved gameplay experiences to train the underlying model.
"""
import gym
from dqn_agent import DqnAgent
from replay_buffer import ReplayBuffer


def collect_gameplay_experiences(env, agent, buffer):
    """
    Collects gameplay experiences by playing env with the instructions
    produced by agent and stores the gameplay experiences in buffer.

    :param env: the game environment
    :param agent: the DQN agent
    :param buffer: the replay buffer
    :return: None
    """
    state = env.reset()
    done = False
    while not done:
        action = agent.policy(state)
        next_state, reward, done, _ = env.step(action)
        buffer.store_gameplay_experience(state, next_state,
                                         reward, action, done)
        state = next_state


def train_model(
        max_episodes=200,
):
    """
    Trains a DQN agent to play the CartPole game by trial and error

    :return: None
    """
    agent = DqnAgent()
    buffer = ReplayBuffer()
    env = gym.make('CartPole-v0')

    for episode_cnt in range(max_episodes):
        collect_gameplay_experiences(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = agent.train(gameplay_experience_batch)
        print('So far the loss is {0}'.format(loss))

    print('No bug lol!!!')


train_model()
