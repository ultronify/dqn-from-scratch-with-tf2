class ReplayBuffer:
    """
    Replay Buffer

    Stores and retrieves gameplay experiences
    """

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
        return

    def sample_gameplay_batch(self):
        """
        Samples a batch of gameplay experiences for training.

        :return: a list of gameplay experiences
        """
        return []
