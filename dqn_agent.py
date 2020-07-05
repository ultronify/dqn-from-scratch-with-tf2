class DqnAgent:
    """
    DQN Agent

    The agent that explores the game and learn how to play the game by
    learning how to predict the expected long-term return, the Q value given
    a state-action pair.
    """

    def policy(self, state):
        """
        Takes a state from the game environment and returns an action that
        has the highest Q value and should be taken as the next step.

        :param state: the current game environment state
        :return: an action
        """
        action = 0  # For now, we will use a 0 action as a placeholder
        return action

    def train(self, batch):
        """
        Trains the underlying network with a batch of gameplay experiences to
        help it better predict the Q values.

        :param batch: a batch of gameplay experiences
        :return: training loss
        """
        loss = 0  # For now, we will just return 0 loss as a placeholder
        return loss
